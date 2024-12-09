from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pprint import pprint

from ..config import EnvConfig
from ..models.state_models import PlanExecute


class AnonymizeQuestion(BaseModel):
    """Anonymized question and mapping."""
    anonymized_question : str = Field(description="Anonymized question.")
    mapping: dict = Field(description="Mapping of original name entities to variables.")
    explanation: str = Field(description="Explanation of the action.")

anonymize_question_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)


anonymize_question_prompt_template = """ You are a question anonymizer. The input You receive is a string containing several words that
 construct a question {question}. Your goal is to changes all name entities in the input to variables, and remember the mapping of the original name entities to the variables.
 ```example1:
        if the input is \"who is harry potter?\" the output should be \"who is X?\" and the mapping should be {{\"X\": \"harry potter\"}} ```
```example2:
        if the input is \"how did the bad guy played with the alex and rony?\"
          the output should be \"how did the X played with the Y and Z?\" and the mapping should be {{\"X\": \"bad guy\", \"Y\": \"alex\", \"Z\": \"rony\"}}```
 you must replace all name entities in the input with variables, and remember the mapping of the original name entities to the variables.
  output the anonymized question and the mapping in a json format. {format_instructions}"""


def init_anonymize_question_chain():
    anonymize_question_prompt = PromptTemplate(
        template=anonymize_question_prompt_template,
    input_variables=["question"],
        partial_variables={"format_instructions": anonymize_question_parser.get_format_instructions()},
    )

    anonymize_question_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )
    
    anonymize_question_chain = anonymize_question_prompt | anonymize_question_llm | anonymize_question_parser
    return anonymize_question_chain

def anonymize_queries(state: PlanExecute):
    """
    Anonymizes the question.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the anonymized question and mapping.
    """
    state.curr_state = "anonymize_question"
    print("Anonymizing question")
    pprint("--------------------")
    anonymize_question_chain = init_anonymize_question_chain()
    anonymized_question_output = anonymize_question_chain.invoke(state.question)
    anonymized_question = anonymized_question_output["anonymized_question"]
    print(f'anonimized_querry: {anonymized_question}')
    pprint("--------------------")
    mapping = anonymized_question_output["mapping"]
    state.anonymized_question = anonymized_question
    state.mapping = mapping
    return state
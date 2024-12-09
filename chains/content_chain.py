from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from ..config.env_config import EnvConfig
from ..utils.helper_functions import escape_quotes
from pprint import pprint

from langchain_core.output_parsers import JsonOutputParser

class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(description="The relevant content from the retrieved documents")

# Prompt templates
keep_relevant_content_template = """you receive a query: {query} and retrieved documents: {retrieved_documents} from a
vector store.
You need to filter out all the non relevant information that don't supply important information regarding the {query}.
your goal is just to filter out the non relevant information.
you can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
output the filtered relevant content.
"""

# Initialize chains
def init_keep_relevant_chain():
    llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )
    keep_relevant_prompt = PromptTemplate(
        template=keep_relevant_content_template,
        input_variables=["query", "retrieved_documents"],
    )
    keep_relevant_chain = keep_relevant_prompt | llm.with_structured_output(KeepRelevantContent)
    return keep_relevant_chain

def keep_only_relevant_content(state):
    """
    Keeps only the relevant content from the retrieved documents that is relevant to the query.

    Args:
        question: The query question.
        context: The retrieved documents.
        chain: The LLMChain instance.

    Returns:
        The relevant content from the retrieved documents that is relevant to the query.
    """
    question = state["question"]
    context = state["context"]

    input_data = {
    "query": question,
    "retrieved_documents": context
}
    print("keeping only the relevant content...")
    pprint("--------------------")
    keep_relevant_chain = init_keep_relevant_chain()
    output = keep_relevant_chain.invoke(input_data)
    relevant_content = output.relevant_content
    relevant_content = "".join(relevant_content)
    relevant_content = escape_quotes(relevant_content)

    return {"relevant_context": relevant_content, "context": context, "question": question}

is_distilled_content_grounded_on_content_prompt_template = """you receive some distilled content: {distilled_content} and the original context: {original_context}.
    you need to determine if the distilled content is grounded on the original context.
    if the distilled content is grounded on the original context, set the grounded field to true.
    if the distilled content is not grounded on the original context, set the grounded field to false. {format_instructions}"""
  

class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(description="Whether the distilled content is grounded on the original context.")
    explanation: str = Field(description="An explanation of why the distilled content is or is not grounded on the original context.")

# Initialize chains
def init_is_distilled_content_grounded_on_content_chain():
    is_distilled_content_grounded_on_content_json_parser = JsonOutputParser(pydantic_object=IsDistilledContentGroundedOnContent)

    is_distilled_content_grounded_on_content_prompt = PromptTemplate(
    template=is_distilled_content_grounded_on_content_prompt_template,
    input_variables=["distilled_content", "original_context"],
        partial_variables={"format_instructions": is_distilled_content_grounded_on_content_json_parser.get_format_instructions()},
    )

    is_distilled_content_grounded_on_content_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )

    is_distilled_content_grounded_on_content_chain = is_distilled_content_grounded_on_content_prompt | is_distilled_content_grounded_on_content_llm | is_distilled_content_grounded_on_content_json_parser

    return is_distilled_content_grounded_on_content_chain


def is_distilled_content_grounded_on_content(state):
    pprint("--------------------")

    """
    Determines if the distilled content is grounded on the original context.

    Args:
        distilled_content: The distilled content.
        original_context: The original context.

    Returns:
        Whether the distilled content is grounded on the original context.
    """

    print("Determining if the distilled content is grounded on the original context...")
    distilled_content = state["relevant_context"]
    original_context = state["context"]

    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context
    }

    is_distilled_content_grounded_on_content_chain = init_is_distilled_content_grounded_on_content_chain()
    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output["grounded"]

    if grounded:
        print("The distilled content is grounded on the original context.")
        return "grounded on the original context"
    else:
        print("The distilled content is not grounded on the original context.")
        return "not grounded on the original context"
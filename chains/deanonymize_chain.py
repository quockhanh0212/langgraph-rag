from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from pprint import pprint

from ..config import EnvConfig
from ..models.state_models import PlanExecute

class DeAnonymizePlan(BaseModel):
    """Possible results of the action."""
    plan: List = Field(description="Plan to follow in future. with all the variables replaced with the mapped words.")

de_anonymize_plan_prompt_template = """ you receive a list of tasks: {plan}, where some of the words are replaced with mapped variables. you also receive
the mapping for those variables to words {mapping}. replace all the variables in the list of tasks with the mapped words. if no variables are present,
return the original list of tasks. in any case, just output the updated list of tasks in a json format as described here, without any additional text apart from the
"""


def init_de_anonymize_plan_chain():
    de_anonymize_plan_prompt = PromptTemplate(
        template=de_anonymize_plan_prompt_template,
        input_variables=["plan", "mapping"],
    )

    de_anonymize_plan_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )
    de_anonymize_plan_chain = de_anonymize_plan_prompt | de_anonymize_plan_llm.with_structured_output(DeAnonymizePlan)
    return de_anonymize_plan_chain

def deanonymize_queries(state: PlanExecute):
    """
    De-anonymizes the plan.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the de-anonymized plan.
    """
    state.curr_state = "de_anonymize_plan"
    print("De-anonymizing plan")
    pprint("--------------------")
    de_anonymize_plan_chain = init_de_anonymize_plan_chain()
    deanonimzed_plan = de_anonymize_plan_chain.invoke({"plan": state.plan, "mapping": state.mapping})
    state.plan = deanonimzed_plan['plan']
    print(f'de-anonimized_plan: {state.plan}')
    return state
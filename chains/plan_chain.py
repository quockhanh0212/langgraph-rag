from pprint import pprint
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers.json import JsonOutputParser

from ..models.state_models import PlanExecute
from ..config import EnvConfig
from ..utils.helper_functions import text_wrap

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


def init_planner():
    planner_prompt =""" For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    """

    planner_prompt = PromptTemplate(
        template=planner_prompt,
        input_variables=["question"],
    )

    planner_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )

    planner = planner_prompt | planner_llm.with_structured_output(Plan)
    return planner

def plan_step(state: PlanExecute):
    """
    Plans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state.curr_state = "planner"
    print("Planning step")
    pprint("--------------------")
    planner = init_planner()
    plan = planner.invoke({"question": state.anonymized_question})
    state.plan = plan['steps']
    print(f'plan: {state.plan}')
    return state


break_down_plan_prompt_template = """You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
you need to go through the plan refine it according to this:
1. every step has to be able to be executed by either:
    i. retrieving relevant information from a vector store of book chunks
    ii. retrieving relevant information from a vector store of chapter summaries
    iii. retrieving relevant information from a vector store of book quotes
    iv. answering a question from a given context.
2. every step should contain all the information needed to execute it.

output the refined plan
"""

def init_break_down_plan_chain():
    break_down_plan_prompt = PromptTemplate(
        template=break_down_plan_prompt_template,
        input_variables=["plan"],
    )

    break_down_plan_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )

    break_down_plan_chain = break_down_plan_prompt | break_down_plan_llm.with_structured_output(Plan)
    return break_down_plan_chain

def break_down_plan_step(state: PlanExecute):
    """
    Breaks down the plan steps into retrievable or answerable tasks.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the refined plan.
    """
    state.curr_state = "break_down_plan"
    print("Breaking down plan steps into retrievable or answerable tasks")
    pprint("--------------------")
    break_down_plan_chain = init_break_down_plan_chain()
    refined_plan = break_down_plan_chain.invoke(state.plan)
    state.plan = refined_plan['steps']
    return state

class ActPossibleResults(BaseModel):
    """Possible results of the action."""
    plan: Plan = Field(description="Plan to follow in future.")
    explanation: str = Field(description="Explanation of the action.")
    

act_possible_results_parser = JsonOutputParser(pydantic_object=ActPossibleResults)

replanner_prompt_template =""" For the given objective, come up with a simple step by step plan of how to figure out the answer. 
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

assume that the answer was not found yet and you need to update the plan accordingly, so the plan should never be empty.

Your objective was this:
{question}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

You already have the following context:
{aggregated_context}

Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.
Do not return previously done steps as part of the plan.

the format is json so escape quotes and new lines.

{format_instructions}

"""

def init_replanner():
    replanner_prompt = PromptTemplate(
        template=replanner_prompt_template,
    input_variables=["question", "plan", "past_steps", "aggregated_context"],
        partial_variables={"format_instructions": act_possible_results_parser.get_format_instructions()},
    )

    replanner_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )

    replanner = replanner_prompt | replanner_llm | act_possible_results_parser
    return replanner

def replan_step(state: PlanExecute):
    """
    Replans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state.curr_state = "replan"
    print("Replanning step")
    pprint("--------------------")
    inputs = {"question": state.question, "plan": state.plan, "past_steps": state.past_steps, "aggregated_context": state.aggregated_context}
    replanner = init_replanner()
    output = replanner.invoke(inputs)
    state.plan = output['plan']['steps']
    return state

class CanBeAnsweredAlready(BaseModel):
    """Possible results of the action."""
    can_be_answered: bool = Field(description="Whether the question can be fully answered or not based on the given context.")

can_be_answered_already_prompt_template = """You receive a query: {question} and a context: {context}.
You need to determine if the question can be fully answered relying only the given context.
The only infomation you have and can rely on is the context you received. 
you have no prior knowledge of the question or the context.
if you think the question can be answered based on the context, output 'true', otherwise output 'false'.
"""

def init_can_be_answered_already_chain():
    can_be_answered_already_prompt = PromptTemplate(
        template=can_be_answered_already_prompt_template,
        input_variables=["question","context"],
    )

    can_be_answered_already_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )
    can_be_answered_already_chain = can_be_answered_already_prompt | can_be_answered_already_llm.with_structured_output(CanBeAnsweredAlready)
    return can_be_answered_already_chain

def can_be_answered(state: PlanExecute):
    """
    Determines if the question can be answered.
    Args:
        state: The current state of the plan execution.
    Returns:
        whether the original question can be answered or not.
    """
    state.curr_state = "can_be_answered_already"
    print("Checking if the ORIGINAL QUESTION can be answered already")
    pprint("--------------------")
    question = state.question
    context = state.aggregated_context
    inputs = {"question": question, "context": context}
    can_be_answered_already_chain = init_can_be_answered_already_chain()
    output = can_be_answered_already_chain.invoke(inputs)
    if output['can_be_answered'] == True:
        print("The ORIGINAL QUESTION can be fully answered already.")
        pprint("--------------------")
        print("the aggregated context is:")
        print(text_wrap(state.aggregated_context))
        print("--------------------")
        return "can_be_answered_already"
    else:
        print("The ORIGINAL QUESTION cannot be fully answered yet.")
        pprint("--------------------")
        return "cannot_be_answered_yet"
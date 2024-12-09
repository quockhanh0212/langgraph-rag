from pprint import pprint
from ..models.state_models import PlanExecute
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from ..config import EnvConfig

tasks_handler_prompt_template = """You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
You have the following tools at your disposal:
Tool A: a tool that retrieves relevant information from a vector store of book chunks based on a given query.
- use Tool A when you think the current task should search for information in the book chunks.
Took B: a tool that retrieves relevant information from a vector store of chapter summaries based on a given query.
- use Tool B when you think the current task should search for information in the chapter summaries.
Tool C: a tool that retrieves relevant information from a vector store of quotes from the book based on a given query.
- use Tool C when you think the current task should search for information in the book quotes.
Tool D: a tool that answers a question from a given context.
- use Tool D ONLY when you the current task can be answered by the aggregated context {aggregated_context}

you also receive the last tool used {last_tool}
if {last_tool} was retrieve_chunks, use other tools than Tool A.

You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
if you decide to use Tools A,B or C, output the query to be used for the tool and also output the relevant tool.
if you decide to use Tool D, output the question to be used for the tool, the context, and also that the tool to be used is Tool D.

"""

class TaskHandlerOutput(BaseModel):
    """Output schema for the task handler."""
    query: str = Field(description="The query to be either retrieved from the vector store, or the question that should be answered from context.")
    curr_context: str = Field(description="The context to be based on in order to answer the query.")
    tool: str = Field(description="The tool to be used should be either retrieve_chunks, retrieve_summaries, retrieve_quotes, or answer_from_context.")

def init_task_handler_chain():
    task_handler_prompt = PromptTemplate(
        template=tasks_handler_prompt_template,
        input_variables=["curr_task", "aggregated_context", "last_tool" "past_steps", "question"],
    )

    task_handler_llm = AzureChatOpenAI(
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
    )
    task_handler_chain = task_handler_prompt | task_handler_llm.with_structured_output(TaskHandlerOutput)
    return task_handler_chain

def run_task_handler_chain(state: PlanExecute):
    """ Run the task handler chain to decide which tool to use to execute the task.
    Args:
       state: The current state of the plan execution.
    Returns:
       The updated state of the plan execution.
    """
    state.curr_state = "task_handler"
    print("the current plan is:")
    print(state.plan)
    pprint("--------------------") 

    if not state.past_steps:
        state.past_steps = []

    curr_task = state.plan[0]

    inputs = {"curr_task": curr_task,
               "aggregated_context": state.aggregated_context,
                "last_tool": state.tool,
                "past_steps": state.past_steps,
                "question": state.question}
    
    task_handler_chain = init_task_handler_chain()
    output = task_handler_chain.invoke(inputs)
  
    state.past_steps.append(curr_task)
    state.plan.pop(0)

    if output['tool'] == "retrieve_chunks":
        state.query_to_retrieve_or_answer = output['query']
        state.tool="retrieve_chunks"
    
    elif output['tool'] == "retrieve_summaries":
        state.query_to_retrieve_or_answer = output['query']
        state.tool="retrieve_summaries"

    elif output['tool'] == "retrieve_quotes":
        state.query_to_retrieve_or_answer = output['query']
        state.tool="retrieve_quotes"

    
    elif output['tool'] == "answer_from_context":
        state.query_to_retrieve_or_answer = output['query']
        state.curr_context = output['curr_context']
        state.tool="answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")
    return state

def retrieve_or_answer(state: PlanExecute):
    """Decide whether to retrieve or answer the question based on the current state.
    Args:
        state: The current state of the plan execution.
    Returns:
        updates the tool to use .
    """
    state.curr_state = "decide_tool"
    print("deciding whether to retrieve or answer")
    if state.tool == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    elif state.tool == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    elif state.tool == "retrieve_quotes":
        return "chosen_tool_is_retrieve_quotes"
    elif state.tool == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")
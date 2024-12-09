from langgraph.graph import END, StateGraph
from typing import TypedDict
from pprint import pprint

from ..models.state_models import QualitativeAnswerGraphState
from ..chains.answer_chain import answer_question_from_context, is_answer_grounded_on_context


qualitative_answer_workflow = StateGraph(QualitativeAnswerGraphState)

# Define the nodes

qualitative_answer_workflow.add_node("answer_question_from_context",answer_question_from_context)

# Build the graph
qualitative_answer_workflow.set_entry_point("answer_question_from_context")

qualitative_answer_workflow.add_conditional_edges(
"answer_question_from_context",is_answer_grounded_on_context ,{"hallucination":"answer_question_from_context", "grounded on context":END}

)

qualitative_answer_workflow_app = qualitative_answer_workflow.compile()

def run_qualtative_answer_workflow(state):
    """
    Run the qualitative answer workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state.curr_state = "answer"
    print("Running the qualitative answer workflow...")
    question = state.query_to_retrieve_or_answer
    context = state.curr_context
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not state.aggregated_context:
        state.aggregated_context = ""
    state.aggregated_context += output["answer"]
    return state

def run_qualtative_answer_workflow_for_final_answer(state):
    """
    Run the qualitative answer workflow for the final answer.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated response.
    """
    state.curr_state = "get_final_answer"
    print("Running the qualitative answer workflow for final answer...")
    question = state.question
    context = state.aggregated_context
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, value in output.items():
            pass  
        pprint("--------------------")
    state.response = value
    return state
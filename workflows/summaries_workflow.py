from langgraph.graph import END, StateGraph
from pprint import pprint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ..models.state_models import QualitativeRetrievalGraphState
from ..chains.content_chain import keep_only_relevant_content, is_distilled_content_grounded_on_content
from ..utils.helper_functions import escape_quotes
from ..utils.vectorstore import encode_chapter_summaries
from ..config import EnvConfig

def create_summaries_query_retriever():
    chapter_summaries = encode_chapter_summaries(EnvConfig.PDF_PATH)
    summaries_query_retriever = chapter_summaries.as_retriever(search_kwargs={"k": 1})
    return summaries_query_retriever

qualitative_summaries_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)

def retrieve_summaries_context_per_question(state):

    print("Retrieving relevant chapter summaries...")
    question = state["question"]
    chapter_summaries_query_retriever = create_summaries_query_retriever()
    docs_summaries = chapter_summaries_query_retriever.get_relevant_documents(state["question"])

    # Concatenate chapter summaries with citation information
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
    )
    context_summaries = escape_quotes(context_summaries)
    return {"context": context_summaries, "question": question}

# Define the nodes
qualitative_summaries_retrieval_workflow.add_node("retrieve_summaries_context_per_question",retrieve_summaries_context_per_question)
qualitative_summaries_retrieval_workflow.add_node("keep_only_relevant_content",keep_only_relevant_content)

# Build the graph
qualitative_summaries_retrieval_workflow.set_entry_point("retrieve_summaries_context_per_question")

qualitative_summaries_retrieval_workflow.add_edge("retrieve_summaries_context_per_question", "keep_only_relevant_content")

qualitative_summaries_retrieval_workflow.add_conditional_edges(
    "keep_only_relevant_content",
    is_distilled_content_grounded_on_content,
    {"grounded on the original context":END,
      "not grounded on the original context":"keep_only_relevant_content"},
    )

def run_qualitative_summaries_retrieval_workflow(state):
    """
    Run the qualitative summaries retrieval workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state.curr_state = "retrieve_summaries"
    print("Running the qualitative summaries retrieval workflow...")
    question = state.query_to_retrieve_or_answer
    inputs = {"question": question}
    qualitative_summaries_retrieval_workflow_app = qualitative_summaries_retrieval_workflow.compile()
    for output in qualitative_summaries_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not state.aggregated_context:
        state.aggregated_context = ""
    state.aggregated_context += output['relevant_context']
    return state

from langgraph.graph import END, StateGraph
from pprint import pprint

from ..models.state_models import QualitativeRetrievalGraphState
from ..chains.content_chain import keep_only_relevant_content, is_distilled_content_grounded_on_content
from ..utils.helper_functions import escape_quotes
from ..utils.vectorstore import encode_book
from ..config import EnvConfig

def create_chunks_query_retriever():
    book_chunks = encode_book(EnvConfig.PDF_PATH, chunk_size=1000, chunk_overlap=200)
    chunks_query_retriever = book_chunks.as_retriever(search_kwargs={"k": 1})
    return chunks_query_retriever

def retrieve_chunks_context_per_question(state):
    """
    Retrieves relevant context for a given question. The context is retrieved from the book chunks and chapter summaries.

    Args:
        state: A dictionary containing the question to answer.
    """
    # Retrieve relevant documents
    print("Retrieving relevant chunks...")
    question = state["question"]
    chunks_query_retriever = create_chunks_query_retriever()
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    context = " ".join(doc.page_content for doc in docs)
    context = escape_quotes(context)
    return {"context": context, "question": question}

qualitative_chunks_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)

# Define the nodes
qualitative_chunks_retrieval_workflow.add_node("retrieve_chunks_context_per_question",retrieve_chunks_context_per_question)
qualitative_chunks_retrieval_workflow.add_node("keep_only_relevant_content",keep_only_relevant_content)

# Build the graph
qualitative_chunks_retrieval_workflow.set_entry_point("retrieve_chunks_context_per_question")

qualitative_chunks_retrieval_workflow.add_edge("retrieve_chunks_context_per_question", "keep_only_relevant_content")

qualitative_chunks_retrieval_workflow.add_conditional_edges(
    "keep_only_relevant_content",
    is_distilled_content_grounded_on_content,
    {"grounded on the original context":END,
      "not grounded on the original context":"keep_only_relevant_content"},
    )

def run_qualitative_chunks_retrieval_workflow(state):
    """
    Run the qualitative chunks retrieval workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state.curr_state = "retrieve_chunks"
    print("Running the qualitative chunks retrieval workflow...")
    question = state.query_to_retrieve_or_answer
    inputs = {"question": question}
    qualitative_chunks_retrieval_workflow_app = qualitative_chunks_retrieval_workflow.compile()
    for output in qualitative_chunks_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not state.aggregated_context:
        state.aggregated_context = ""
    state.aggregated_context += output['relevant_context']
    return state
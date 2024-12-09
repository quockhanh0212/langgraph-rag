from langgraph.graph import END, StateGraph
from pprint import pprint
from ..models.state_models import QualitativeRetrievalGraphState
from ..utils.vectorstore import encode_quotes
from ..utils.helper_functions import escape_quotes
from ..chains.content_chain import keep_only_relevant_content, is_distilled_content_grounded_on_content
from ..config import EnvConfig

def create_quotes_query_retriever():
    book_quotes = encode_quotes(EnvConfig.PDF_PATH)
    quotes_query_retriever = book_quotes.as_retriever(search_kwargs={"k": 10})
    return quotes_query_retriever

def retrieve_book_quotes_context_per_question(state):
    question = state["question"]

    print("Retrieving relevant book quotes...")
    book_quotes_query_retriever = create_quotes_query_retriever()
    docs_book_quotes = book_quotes_query_retriever.get_relevant_documents(state["question"])
    book_qoutes = " ".join(doc.page_content for doc in docs_book_quotes)
    book_qoutes_context = escape_quotes(book_qoutes)

    return {"context": book_qoutes_context, "question": question}

qualitative_book_quotes_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)

# Define the nodes
qualitative_book_quotes_retrieval_workflow.add_node("retrieve_book_quotes_context_per_question",retrieve_book_quotes_context_per_question)
qualitative_book_quotes_retrieval_workflow.add_node("keep_only_relevant_content",keep_only_relevant_content)

# Build the graph
qualitative_book_quotes_retrieval_workflow.set_entry_point("retrieve_book_quotes_context_per_question")

qualitative_book_quotes_retrieval_workflow.add_edge("retrieve_book_quotes_context_per_question", "keep_only_relevant_content")

qualitative_book_quotes_retrieval_workflow.add_conditional_edges(
    "keep_only_relevant_content",
    is_distilled_content_grounded_on_content,
    {"grounded on the original context":END,
      "not grounded on the original context":"keep_only_relevant_content"},
    )

qualitative_book_quotes_retrieval_workflow_app = qualitative_book_quotes_retrieval_workflow.compile()

def run_qualitative_book_quotes_retrieval_workflow(state):
    """
    Run the qualitative book quotes retrieval workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state.curr_state = "retrieve_book_quotes"
    print("Running the qualitative book quotes retrieval workflow...")
    question = state.query_to_retrieve_or_answer
    inputs = {"question": question}
    for output in qualitative_book_quotes_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not state.aggregated_context:
        state.aggregated_context = ""
    state.aggregated_context += output['relevant_context']
    return state
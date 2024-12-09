from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from ..utils.helper_functions import replace_t_with_space, num_tokens_from_string, replace_double_lines_with_one_line, split_into_chapters, extract_book_quotes_as_documents

from time import monotonic

from ..config import EnvConfig


MODEL_NAME = "avsolatorio/GIST-large-Embedding-v0"

summarization_prompt_template = """Write an extensive summary of the following:

{text}

SUMMARY:"""

summarization_prompt = PromptTemplate(template=summarization_prompt_template, input_variables=["text"])

def encode_book(path, chunk_size=1000, chunk_overlap=200):
    """Encodes a PDF book into a vector store."""
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

def create_chapters(chapter_path):
    chapters = split_into_chapters(chapter_path) 
    chapters = replace_t_with_space(chapters)
    return chapters

def create_chapter_summary(chapter):
    """
    Creates a summary of a chapter using a large language model (LLM).

    Args:
        chapter: A Document object representing the chapter to summarize.

    Returns:
        A Document object containing the summary of the chapter.
    """

    chapter_txt = chapter.page_content  # Extract chapter text
    model_name = "gpt-4o"  # Specify LLM model
    llm = AzureChatOpenAI(
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
    )  # Create LLM instance
    gpt_4o_max_tokens = 128000  # Maximum token limit for the LLM
    verbose = False  # Set to True for more detailed output

    # Calculate number of tokens in the chapter text
    num_tokens = num_tokens_from_string(chapter_txt, model_name)

    # Choose appropriate chain type based on token count
    if num_tokens < gpt_4o_max_tokens:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=summarization_prompt, verbose=verbose) 
    else:
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=summarization_prompt, combine_prompt=summarization_prompt, verbose=verbose)

    start_time = monotonic()  # Start timer
    doc_chapter = Document(page_content=chapter_txt)  # Create Document object for chapter
    summary = chain.invoke([doc_chapter])  # Generate summary using the chain
    print(f"Chain type: {chain.__class__.__name__}")  # Print chain type
    print(f"Run time: {monotonic() - start_time}")  # Print execution time

    # Clean up summary text
    summary = replace_double_lines_with_one_line(summary["output_text"])

    # Create Document object for summary
    doc_summary = Document(page_content=summary, metadata=chapter.metadata)

    return doc_summary

def encode_chapter_summaries(chapter_path):
    """Encodes chapter summaries into a vector store."""
    chapters = create_chapters(chapter_path)
    chapter_summaries = [create_chapter_summary(chapter) for chapter in chapters]
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return FAISS.from_documents(chapter_summaries, embeddings)

def create_book_quotes(book_path):
    loader = PyPDFLoader(book_path)
    document = loader.load()
    document_cleaned = replace_t_with_space(document)
    book_quotes = extract_book_quotes_as_documents(document_cleaned)
    return book_quotes

def encode_quotes(book_path):
    """Encodes book quotes into a vector store."""
    book_quotes = create_book_quotes(book_path)
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return FAISS.from_documents(book_quotes, embeddings)

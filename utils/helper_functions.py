from typing import List
from langchain.docstore.document import Document
import tiktoken
import re
import PyPDF2

def replace_t_with_space(documents: List[Document]) -> List[Document]:
    """Replace 't' characters with spaces in document content."""
    cleaned_docs = []
    for doc in documents:
        cleaned_content = doc.page_content.replace('\t', ' ')
        cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs


def replace_double_lines_with_one_line(text: str) -> str:
    """Replace double line breaks with single line breaks."""
    return text.replace('\n\n', '\n')


def escape_quotes(text: str) -> str:
    """Escape quotes in text."""
    return text.replace('"', '\\"')


def text_wrap(text: str, width: int = 100) -> str:
    """Wrap text to specified width."""
    import textwrap
    return textwrap.fill(text, width=width)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a given string using a specified encoding.

    Args:
        string: The input string to tokenize.
        encoding_name: The name of the encoding to use (e.g., 'cl100k_base').

    Returns:
        The number of tokens in the string according to the specified encoding.
    """

    encoding = tiktoken.encoding_for_model(encoding_name)  # Get the encoding object
    num_tokens = len(encoding.encode(string))  # Encode the string and count tokens
    return num_tokens

def split_into_chapters(book_path):
    """
    Splits a PDF book into chapters based on chapter title patterns.

    Args:
        book_path (str): The path to the PDF book file.

    Returns:
        list: A list of Document objects, each representing a chapter with its text content and chapter number metadata.
    """

    with open(book_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        documents = pdf_reader.pages  # Get all pages from the PDF

        # Concatenate text from all pages
        text = " ".join([doc.extract_text() for doc in documents])

        # Split text into chapters based on chapter title pattern (adjust as needed)
        chapters = re.split(r'(CHAPTER\s[A-Z]+(?:\s[A-Z]+)*)', text)

        # Create Document objects with chapter metadata
        chapter_docs = []
        chapter_num = 1
        for i in range(1, len(chapters), 2):
            chapter_text = chapters[i] + chapters[i + 1]  # Combine chapter title and content
            doc = Document(page_content=chapter_text, metadata={"chapter": chapter_num})
            chapter_docs.append(doc)
            chapter_num += 1

    return chapter_docs

def extract_book_quotes_as_documents(documents, min_length=50):
    quotes_as_documents = []
    # Correct pattern for quotes longer than min_length characters, including line breaks
    quote_pattern_longer_than_min_length = re.compile(rf'“(.{{{min_length},}}?)”', re.DOTALL)

    for doc in documents:
        content = doc.page_content
        content = content.replace('\n', ' ')
        found_quotes = quote_pattern_longer_than_min_length.findall(content)
        for quote in found_quotes:
            quote_doc = Document(page_content=quote)
            quotes_as_documents.append(quote_doc)
    
    return quotes_as_documents


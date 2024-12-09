# LangGraph-RAG

LangGraph-RAG is a Python library designed to facilitate the retrieval and generation of answers from a variety of contexts using language models. It leverages Azure's OpenAI services to process and analyze text data, providing a structured approach to handling complex queries.

## Features

- **Task Handling**: Decides which tool to use for executing tasks based on the context and past steps.
- **Anonymization**: Anonymizes questions by replacing named entities with variables.
- **Content Filtering**: Filters out non-relevant information from retrieved documents.
- **Question Answering**: Provides answers based on a given context using chain-of-thought reasoning.
- **Plan Execution**: Generates and refines plans to achieve specific objectives.

## Installation

To install the necessary dependencies, ensure you have Python installed and run:

```bash
pip install -r requirements.txt
```


## Configuration

The application uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:
```plaintext
AZ_OAI_BASE='your_azure_openai_base_url'
AZ_OPENAI_API_KEY='your_openai_api_key'
AZ_OAI_VERSION='your_azure_openai_version'
AZ_OAI_DEPLOYMENT='your_azure_openai_deployment'
PDF_PATH='path_to_your_pdf_file'
```

## Usage

Run the main.py file to see the output.

```bash
python -m langgraph_rag.main
```

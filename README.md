# LangGraph-RAG

LangGraph-RAG is a Python library designed to facilitate the retrieval and generation of answers from a variety of contexts using language models. It leverages Azure's OpenAI services to process and analyze text data, providing a structured approach to handling complex queries.

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


# HustleHub

HustleHub is a research tool that makes finding information easy. Just enter article URLs and ask questions to get quick insights.


## Features

- Easily load URLs or upload text files with URLs to fetch and analyze article content.
- Utilize LangChain's UnstructuredURL Loader to seamlessly process and extract information from articles.
- Create embedding vectors using OpenAI's embeddings and leverage FAISS for fast and accurate retrieval of relevant insights.
- Engage with the tool by asking questions and receiving answers, complete with source URLs for reference.

## Installation

1. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
2. Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY=your_api_key_here
```
## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main.py
```

## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.
import os
import streamlit as st
import pickle
import time
from bs4 import BeautifulSoup
import requests
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="HustleHub", page_icon="📈", layout="wide")
st.title("HustleHub - 📈")
st.subheader("Quick Research at Your Fingertips.")
st.sidebar.title("Article URLs")

# Collect URLs from user
urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

# Main content placeholder
main_placeholder = st.empty()

# Set up LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Fetch and process data from URLs
    data = []
    for url in urls:
        if url:  # Check if the URL is not empty
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all(['p', 'div', 'article'])
                text = "\n\n".join([para.get_text() for para in paragraphs])

                # Ensure the content is long enough
                if len(text.split()) > 300:
                    document = Document(page_content=text, metadata={"source": url})
                    data.append(document)
            except requests.exceptions.MissingSchema:
                st.error(f"Invalid URL provided: {url}")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while fetching the URL: {url}\n{e}")

    if not data:
        st.error("No data was loaded from the provided URLs.")
    else:
        # Split data into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=3000
        )
        main_placeholder.success("Text Splitting...Completed ✅")
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("No documents were loaded. Please check the URLs and try again.")
        else:
            # Create embeddings and build FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.success("Embeddings & Vectorstore Building...Completed ✅")
            time.sleep(2)

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)
            main_placeholder.success("Vectorstore saved successfully! ✅")

# Query input and result display
query = main_placeholder.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as f:
            try:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources")
                    for source in sources.split("\n"):
                        st.write(source)
            except EOFError:
                st.error("Failed to load the vectorstore. The file might be corrupted or empty.")
    else:
        st.error("No vectorstore file found. Please ensure the URLs are processed first.")
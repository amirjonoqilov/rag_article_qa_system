import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Project", layout="wide")
st.title("RAG Project Interface")
st.sidebar.title("Article URLs")

# -------------------- URL INPUT --------------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Enter article URL {i+1}")
    if url:
        urls.append(url)

process_button = st.sidebar.button("Process Articles")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# -------------------- PROCESS ARTICLES --------------------
if process_button and urls:
    try:
        with st.spinner("Loading articles..."):
            loader = UnstructuredURLLoader(urls=urls, headers=headers)
            documents = loader.load()
            st.success(f"✓ Loaded {len(documents)} documents.")

        with st.spinner("Splitting documents..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(documents)
            st.success(f"✓ Split into {len(documents)} chunks.")

        with st.spinner("Creating embeddings and FAISS index..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local("faiss_index")
            st.session_state.vectorstore_ready = True
            st.success("✓ FAISS index created and saved.")

    except Exception as e:
        st.error(f"Error processing articles: {str(e)}")

# -------------------- QUERY --------------------
st.divider()
st.subheader("Ask a Question")

query = st.text_input("Ask a question about the articles")

if query:
    if os.path.exists("faiss_index"):
        try:
            with st.spinner("Searching for relevant documents..."):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local(
                    "faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                docs = vectorstore.similarity_search(query, k=3)

            context = "\n\n".join(doc.page_content for doc in docs)

            with st.spinner("Generating answer..."):
                ollama = ChatOllama(model="llama3.2", temperature=0)
                prompt = f"""Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""
                response = ollama.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Document {i}:**")
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    else:
        st.warning("Please process articles first using the sidebar.")
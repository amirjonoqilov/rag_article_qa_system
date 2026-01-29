# News Article RAG Question Answering System

This project is a web-based Retrieval-Augmented Generation (RAG) application that allows users to input online article URLs and ask natural language questions about their content.

The system retrieves relevant document chunks using semantic search and generates context-aware answers using a Large Language Model (LLM).

## Features
- Load and process online articles from URLs
- Automatic text chunking and preprocessing
- Semantic embeddings using Hugging Face models
- FAISS vector database for fast similarity search
- Context-aware question answering using LLaMA via Ollama
- Interactive web interface built with Streamlit

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Ollama (LLaMA)
- Retrieval-Augmented Generation (RAG)

## How It Works
1. User inputs article URLs
2. Articles are loaded and split into chunks
3. Text embeddings are generated
4. Chunks are indexed in FAISS
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates an answer using retrieved context

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py

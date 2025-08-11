# RAG Agent with LangGraph and Streamlit
---
This project is a simple Retrieval-Augmented Generation (RAG) application built with LangGraph and Streamlit.
It allows you to upload a PDF, provide your OpenAI API key, and ask questions based on the document content.
---

**Features**
- Interactive Web UI powered by Streamlit.

- Upload any PDF and instantly index its content.

- Dynamic OpenAI API key input (no need to hardcode it).

- LangGraph-powered RAG pipeline for modularity.

- FAISS in-memory vector store for fast retrieval.

---

**Installation**
```bash
1. Clone the repository
```bash
git clone https://github.com/Viswa-Prakash/langgraph_simple_rag_agent.git
cd langgraph_simple_rag_agent

2. Install dependencies
```bash
pip install -r requirements.txt

3. Usage
```bash
streamlit run app.py
```
---
**How It Works**  
1. Enter your OpenAI API Key in the provided field.

2. Upload a PDF document.

3. The app:

- Loads and processes the document.

- Embeds it using OpenAIEmbeddings.

- Stores embeddings in an in-memory FAISS index.

4. Ask a question related to the document, and the RAG agent retrieves relevant context and generates an answer.

![alt text](C:\Users\viswa\Documents\GENAIandAGENTICAI\langgraph_simple_rag_agent\basic-rag-flow.bfbjOors_Z6l3c3.webp)

---
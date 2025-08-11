import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import tempfile
import os

# ---- Streamlit UI ----
st.set_page_config(page_title="Simple RAG Agent", layout="centered")

st.title("RAG Agent with LangGraph")
st.write("Upload a PDF, provide your OpenAI API key, and ask questions based on its content.")


# Step 1: API Key input
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Step 2: PDF upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is None:
    st.warning("Please upload a PDF to continue.")
    st.stop()

# Save uploaded PDF to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
    temp_pdf.write(uploaded_file.read())
    temp_pdf_path = temp_pdf.name

# Step 3: Load PDF
loader = PyPDFLoader(temp_pdf_path)
docs = loader.load()
st.success(f"Loaded {len(docs)} pages from the PDF.")

# Step 4: Create vector store in memory
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.from_documents(docs, embeddings)
retriever = vectordb.as_retriever()

# Step 5: Create LLM + RetrievalQA
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

# Step 6: LangGraph setup
class State(TypedDict):
    question: str
    answer: str

def rag_node(state: State) -> State:
    result = qa_chain({"query": state["question"]})
    return {"question": state["question"], "answer": result["result"]}

graph = StateGraph(State)
graph.add_node("rag", rag_node)
graph.add_edge(START, "rag")
graph.add_edge("rag", END)
rag_agent = graph.compile()

# Step 7: Ask a question
query = st.text_input("Enter your question based on the uploaded document:")

if st.button("Ask"):
    if query.strip():
        state = {"question": query, "answer": ""}
        result = rag_agent.invoke(state)
        st.subheader("Answer:")
        st.write(result["answer"])
    else:
        st.warning("Please enter a question.")


import streamlit as st
import os
import tempfile
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import bs4
import google.generativeai as genai

# Load environment variables
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

if not google_api_key:
    st.error("Google Gemini API key is missing. Please set 'GOOGLE_API_KEY' in the environment variables.")
    st.stop()

# Initialize Google Gemini Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Function to load text files
def load_text_file(file):
    loader = TextLoader(file)
    return loader.load()

# Function to load PDFs (with temporary file handling)
def load_pdf_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())  # Save file temporarily
        temp_path = temp_file.name
    
    loader = PyMuPDFLoader(temp_path)
    return loader.load()

# Function to load web content
def load_web_content(url):
    loader = WebBaseLoader(web_paths=(url,),
                           bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                               class_=("post-title", "post-content", "post-header"))))
    return loader.load()

# Function to create FAISS index from documents
def create_faiss_index(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Generate embeddings and store in FAISS
    vector_db = FAISS.from_documents(chunks, embedding_model)
    return vector_db

# Function to retrieve relevant docs using FAISS
def retrieve_relevant_docs(query, vector_db, top_k=3):
    docs = vector_db.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

# Function to generate response using Google Gemini
def get_llm_response(query, context):
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel("models/gemini-1.5-pro")
    
    prompt = f"Use the following context to answer the query.\n\nContext:\n{context}\n\nQuery: {query}"
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("🔍 RAG-based QnA Chatbot with FAISS")

# File or URL input
option = st.radio("Choose Data Source:", ("Upload Text File", "Upload PDF File", "Enter Website URL"))

documents = None
if option == "Upload Text File":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        documents = load_text_file(uploaded_file)
        st.success("✅ Text file loaded successfully!")

elif option == "Upload PDF File":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        documents = load_pdf_file(uploaded_pdf)
        st.success("✅ PDF file loaded successfully!")

elif option == "Enter Website URL":
    url = st.text_input("Enter website URL:")
    if st.button("Load Web Content") and url:
        documents = load_web_content(url)
        st.success("✅ Web content loaded successfully!")

# Check if documents are loaded
if documents:
    st.subheader("📄 Extracted Content")
    st.text_area("Processed Text", documents[0].page_content[:1000], height=200)

    # Create FAISS Index
    st.write("🔄 Indexing document...")
    faiss_index = create_faiss_index(documents)
    st.success("✅ Document indexed successfully!")

    # QnA Section
    query = st.text_input("📝 Ask a question:")
    if st.button("Get Answer") and query:
        # Retrieve relevant docs using FAISS
        relevant_context = retrieve_relevant_docs(query, faiss_index)
        
        # Generate response using Gemini
        answer = get_llm_response(query, relevant_context)
        
        # Display response
        st.subheader("💡 Answer:")
        st.write(answer)

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
    st.error("❌ Google Gemini API key is missing. Please set 'GOOGLE_API_KEY' in the environment variables.")
    st.stop()

# Initialize Google Gemini Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Streamlit Page Config with Dark Theme
st.set_page_config(page_title="AI-Powered QnA Chatbot", page_icon="🤖", layout="wide")

# Inject Custom CSS for Full Dark Theme
st.markdown(
    """
    <style>
    /* Background and text color */
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #161A23 !important;
    }

    /* Style text input fields */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: #1F2937 !important;
        color: white !important;
        border-radius: 5px;
    }

    /* Style buttons */
    .stButton > button {
        background-color: #F63366 !important;
        color: white !important;
        border-radius: 5px;
        width: 100%;
    }

    /* Style success and warning messages */
    .stAlert {
        background-color: #1E2A38 !important;
        color: white !important;
    }

    /* File uploader styling */
    .stFileUploader > div {
        background-color: #1F2937 !important;
        color: white !important;
        border-radius: 5px;
    }

    /* Expander Styling */
    .stExpander > summary {
        background-color: #1F2937 !important;
        color: white !important;
        border-radius: 5px;
    }

    /* Centering footer */
    .footer {
        text-align: center;
        color: grey;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.title("🔍 AI-Powered QnA Chatbot with FAISS")
st.markdown("🚀 Ask questions from documents, websites, or PDFs!")

# Function to load text files
def load_text_file(file):
    loader = TextLoader(file)
    return loader.load()

# Function to load PDFs (with temporary file handling)
def load_pdf_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
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

# Sidebar for File or URL input
st.sidebar.title("📂 Data Source")
option = st.sidebar.radio("Choose Source:", ("Upload Text File", "Upload PDF File", "Enter Website URL"))

documents = None
if option == "Upload Text File":
    uploaded_file = st.sidebar.file_uploader("📄 Upload a text file", type=["txt"])
    if uploaded_file:
        documents = load_text_file(uploaded_file)
        st.sidebar.success("✅ Text file loaded successfully!")

elif option == "Upload PDF File":
    uploaded_pdf = st.sidebar.file_uploader("📄 Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        documents = load_pdf_file(uploaded_pdf)
        st.sidebar.success("✅ PDF file loaded successfully!")

elif option == "Enter Website URL":
    url = st.sidebar.text_input("🌐 Enter website URL:")
    if st.sidebar.button("Load Web Content") and url:
        documents = load_web_content(url)
        st.sidebar.success("✅ Web content loaded successfully!")

# Check if documents are loaded
if documents:
    with st.expander("📄 Extracted Content", expanded=False):
        st.text_area("Processed Text", documents[0].page_content[:1000], height=150)
    
    # Create FAISS Index with progress
    with st.spinner("🔄 Indexing document..."):
        faiss_index = create_faiss_index(documents)
    st.success("✅ Document indexed successfully!")

    # QnA Section
    query = st.text_input("📝 Ask a question:")
    if st.button("Get Answer", use_container_width=True):
        if query:
            with st.spinner("🔍 Searching for relevant information..."):
                relevant_context = retrieve_relevant_docs(query, faiss_index)
                answer = get_llm_response(query, relevant_context)
            
            # Display response
            st.subheader("💡 Answer:")
            st.write(answer)
            
            # Rating Section
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct Answer", use_container_width=True):
                    st.success("✅ Feedback recorded. Thank you!")
            with col2:
                if st.button("👎 Incorrect Answer", use_container_width=True):
                    st.warning("⚠️ Feedback noted. Please provide a better answer in feedback section.")
            
            # User Feedback Text Box
            feedback = st.text_area("✍️ Suggest a better answer (if incorrect):", "", height=100)
            if st.button("Submit Feedback") and feedback:
                with open("feedback_data.json", "a") as f:
                    f.write(f"{{'query': '{query}', 'incorrect_answer': '{answer}', 'corrected_answer': '{feedback}'}}\n")
                st.success("✅ Feedback saved! This will be used for future fine-tuning.")

# Footer
st.markdown('<p class="footer">Made with ❤️ using Streamlit</p>', unsafe_allow_html=True)

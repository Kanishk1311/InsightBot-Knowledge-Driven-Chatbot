# InsightBot – Context-Aware Q&A Chatbot

**InsightBot** is a multi-format Retrieval-Augmented Generation (RAG)-based Q&A chatbot built using Streamlit, FAISS, and Google Gemini. It allows users to upload documents or input website URLs and ask context-based questions using semantic search and LLM-powered answers.

---

## 🚀 Features

- **Multi-Format Input**: Accepts PDF, TXT, and Web URL sources.
- **Semantic Search**: Uses FAISS to retrieve relevant content chunks.
- **LLM Responses**: Generates intelligent answers using Google Gemini.
- **Feedback Mode**: Includes a feedback version for collecting user responses.
- **Real-Time Parsing**: Automatically processes and indexes content on upload.

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Embeddings & LLM**: Google Generative AI (Gemini)
- **Vector Store**: FAISS
- **Text Processing**: LangChain (Loaders, Splitters)
- **Environment Management**: dotenv

---

---

## 📦 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/insightbot.git
cd insightbot

2. **Install dependencies**
    pip install -r requirements1.txt
3. Set your Google API key
Create a .env file in the root directory and add:
GOOGLE_API_KEY=your_google_api_key
Running the App
4. Run the version without feedback:
streamlit run app-no-feedback.py
5. Run the version with feedback:
streamlit run app-feedback.py

Example Workflow
Upload a .pdf or .txt file or enter a website URL.

The app extracts, chunks, embeds, and indexes the content.

Ask a question.

InsightBot fetches relevant context and answers using Gemini LLM.

📄 License
This project is licensed under the MIT License.

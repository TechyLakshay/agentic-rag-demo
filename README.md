# 🤖 Agentic RAG Demo

This is a **demo version** of an Agentic RAG (Retrieval-Augmented Generation) system built with:
- Streamlit (UI)
- LangChain (agent + RAG)
- FAISS (vector database)
- Google Generative AI (LLM)
- Tavily Search (web tool)

The agent decides when to use **local knowledge (RAG)** vs **web search** 🌐.

---

## 🚀 Features
- Local document Q&A using FAISS + HuggingFace embeddings
- Web search integration via Tavily
- Simple Streamlit UI
- Modular tools for future expansion

---

## 🛠️ Setup

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/agentic-rag-demo.git
   cd agentic-rag-demo

2. Install modules
  pip install -r requirements.txt


3. Run the App in the Terminal
  streamlit run app.py

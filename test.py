import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise ValueError("Please set the TAVILY_API_KEY environment variable.")

# Page Config
st.set_page_config(page_title="Agentic RAG Demo", page_icon="🤖")
st.title("🤖 Agentic RAG System Demo")
st.write("Ask me questions! I’ll decide when to use **local RAG** vs **Web Search** 🌐")

# --- Setup components (load once) ---
@st.cache_resource
def load_agent():
    # Embeddings + FAISS retriever
    db = FAISS.load_local(
        "attention_pdf_index",
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=api_key
    )

    # RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Wrap tools
    rag_tool = Tool(
        name="rag_retriever",
        func=lambda q: str(qa.invoke(q).get("result", "No result found")),

        description="Use this tool to answer questions from the local knowledge base."
    )

    web_tool = Tool(
        name="Search",
        func=TavilySearchResults().run,
        description="Useful for current events or web-based information and also answer in detail."
    )

    # Initialize Agent
    agent = initialize_agent(
        tools=[rag_tool, web_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
        
    )

    return agent

agent = load_agent()

# --- UI Input ---
user_query = st.text_input("🔎 Ask your question:")

if st.button("Run Agent") and user_query:
    with st.spinner("🤔 Agent is thinking..."):
        try:
            response = agent.run(user_query)
            st.success("✅ Agent Response:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

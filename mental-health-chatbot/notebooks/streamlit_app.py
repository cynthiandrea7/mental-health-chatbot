"""
MongoDB Vector Search Chatbot (Mental Wellness Assistant) with Streamlit and LangChain
"""

# Standard library imports
import os
import sys
import time
from operator import itemgetter
import urllib.parse # <-- ADDED for safe password handling

# Third-party imports
import streamlit as st
from pymongo import MongoClient

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# --- ENVIRONMENT & CONFIGURATION ---

# MongoDB connection details
DB_NAME = "mental_health_db"
COLLECTION_NAME = "support_data"
VECTOR_INDEX_NAME = "vector_index_mental_health"
EMBEDDING_MODEL = "models/gemini-embedding-001" 
LLM_MODEL = "gemini-2.5-flash"

# Load MongoDB components from environment variables (Render/OS environment variables)
from pymongo import MongoClient # import mongo client to connect
import json # import json to load credentials
import urllib.parse

# load credentials from json file
with open('credentials_mongodb.json') as f:
    login = json.load(f)

username = login['username']
password = urllib.parse.quote(login['password'])
host = login['host']

print("MONGO_USER:", username)
# Note: Ensure these are set in the Render environment variables section:
# GOOGLE_API_KEY, MONGO_USER, MONGO_PASS, MONGO_HOST, LANGCHAIN_API_KEY (optional for LangSmith)

if not all([username, password, host, os.environ.get('GOOGLE_API_KEY')]):
    st.error("Missing required MongoDB connection components (MONGO_USER, MONGO_PASS, MONGO_HOST) or GOOGLE_API_KEY. Please configure them in your Render dashboard.")
    st.stop()

# Assemble the MONGO_URI from components (using URL-encoded password)
# MONGO_PASS_QUOTED = urllib.parse.quote(MONGO_PASS)
MONGO_URI = "mongodb+srv://{}:{}@{}/?retryWrites=true&w=majority".format(username, password, host)



# --- CACHED RESOURCE FUNCTIONS ---

@st.cache_resource
def get_mongodb_collection():
    """Connect to MongoDB and return collection (cached)."""
    try:
        # Use the assembled MONGO_URI
        client = MongoClient(MONGO_URI)
        database = client[DB_NAME]
        return database[COLLECTION_NAME]
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        st.stop()


@st.cache_resource
def get_embeddings_and_llm():
    """Initialize embeddings and LLM models (cached)."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        llm = init_chat_model(LLM_MODEL, model_provider="google_genai")
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing LLM/Embeddings: {e}")
        st.stop()


@st.cache_resource
def get_retriever():
    """Initialize vector store retriever (cached)."""
    collection = get_mongodb_collection()
    embeddings, _ = get_embeddings_and_llm()
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
        relevance_score_fn="cosine"
    )
    return vector_store.as_retriever(search_kwargs={'k': 5})

# --- INITIALIZATION ---

# Initialize cached resources
try:
    collection = get_mongodb_collection()
    embeddings, llm = get_embeddings_and_llm()
    retriever = get_retriever()
except Exception as e:
    st.exception(f"Initial setup failed: {e}")
    st.stop()

# Helper function
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- LANGCHAIN RAG CHAIN DEFINITION ---
def get_response(user_query, chat_history):
    
    # **CRITICAL SAFETY PROMPT**
    template = """You are a supportive, friendly AI assistant that helps answer general questions about mental wellness based ONLY on the provided context.
    Your persona is that of an empathetic 'bestie' (use gen-Z slang and emojis sparingly and appropriately to be approachable).
    
    **SAFETY RULE: YOU MUST NOT provide medical advice, diagnosis, or treatment recommendations.**
    If a user expresses a crisis (e.g., suicidal thoughts, severe distress) or asks for immediate medical help, you must gently redirect them to professional crisis resources and stop generating content. For example: "Hey, that sounds really tough. Please reach out to a professional mental health service or crisis hotline for immediate help—they are the real MVPs here."
    
    If the context does not contain the answer, state that the information is not in your current knowledge base, but offer general encouragement.
    
    Context (Retrieved Mental Health Information): {context}
    Chat history: {chat_history}
    User question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # LangChain RAG Chain
    chain = (
        {
            # 1. Retrieve context based on the user's question
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        # 2. Pass context, question, and history to the prompt
        | prompt
        # 3. Generate response using the LLM
        | llm
        # 4. Parse output to a string
        | StrOutputParser()
    )
    
    return chain.stream({
        "chat_history": chat_history,
        "question": user_query,
    })

# --- STREAMLIT UI ---

# Streamlit app config
st.set_page_config(
    page_title="AI Wellness Bestie 2025",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling (as provided by user)
st.markdown("""
    <style>
    :root {
        --ink: #111111;
        --bg: #f7f7f7;
        --ai: #E9D5FF;
        --ai-ink: #00C2FF;
        --user: #FFF0D6;
        --user-ink: #FF8A00;
        --accent: #FFE500;
        --accent-2: #00C2FF;
    }

    /* App background */
    .stApp { background: var(--bg); }

    /* Chat messages */
    .stChatMessage {
        background: #ffffff !important;
        border: 3px solid var(--ink);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.75rem 0;
        box-shadow: 8px 8px 0 var(--ink);
    }

    /* AI and Human variations */
    .stChatMessage[data-testid*="ai"] {
        background: var(--ai) !important;
        border-color: var(--ai-ink);
        box-shadow: 8px 8px 0 var(--ai-ink);
    }
    
    .stChatMessage[data-testid*="user"] {
        background: var(--user) !important;
        border-color: var(--user-ink);
        box-shadow: 8px 8px 0 var(--user-ink);
    }

    /* Input */
    .stChatInputContainer {
        background: #fff;
        border-top: 0;
        padding-top: 1rem;
    }
    
    .stChatInputContainer textarea {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 3px solid var(--ink) !important;
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--ink);
    }
    
    .stChatInputContainer textarea:focus {
        outline: none !important;
        border-color: var(--accent-2) !important;
        box-shadow: 6px 6px 0 var(--accent-2);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--accent);
        border-left: 3px solid var(--ink);
        box-shadow: -8px 0 0 var(--ink);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    /* Buttons */
    .stButton button {
        background: var(--accent-2) !important;
        color: var(--ink) !important;
        border: 3px solid var(--ink) !important;
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--ink);
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    
    .stButton button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0 var(--ink);
    }

    /* Titles */
    h1 {
        color: var(--ink);
        text-align: left;
        font-weight: 900;
        display: inline-block;
        background: var(--accent);
        padding: 6px 12px;
        border: 3px solid var(--ink);
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--ink);
    }
    
    h3 {
        color: #333;
        text-align: left;
        font-weight: 700;
    }

    /* Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: var(--ink) !important;
    }

    /* Divider */
    hr {
        border: 0;
        height: 3px;
        background: var(--ink);
        box-shadow: 4px 4px 0 #ffd400;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 AI Wellness Bestie Assistant")
    st.markdown("---")
    
    st.markdown("#### ⚠️ Important Note")
    st.markdown("""
    This chatbot provides general information for mental wellness only. **It is NOT a substitute for professional medical advice, diagnosis, or treatment.** If you are in crisis, please seek professional help immediately.
    """)
    st.markdown("---")
    
    st.markdown("#### 📊 Session Info")
    if "chat_history" in st.session_state:
        msg_count = len([m for m in st.session_state.chat_history if isinstance(m, HumanMessage)])
        st.metric("Messages Sent", msg_count)
    
    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your AI bestie assistant. How can I help you today?")
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("#### 💡 About")
    st.markdown(f"""
    This chatbot uses:
    - 🔍 MongoDB Atlas Vector Search ({COLLECTION_NAME})
    - 🤖 Google {LLM_MODEL} AI
    - 🔗 LangChain RAG
    """)
    
    st.markdown("---")
    st.markdown("##### Built for Render Deployment")

# Main header
st.title("🧠 AI Wellness Bestie")
st.markdown("### Ask me anything about mental wellness!")
st.markdown("---")


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hey, what's up? I'm here to chat about general mental wellness based on the info I have. Let me know what you're thinking about!"),
    ]

# Display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # The Streamlit `st.write_stream` handles the iterative output from LangChain's .stream()
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))
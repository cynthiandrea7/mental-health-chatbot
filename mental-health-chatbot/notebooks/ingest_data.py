import os
import pandas as pd
import glob
import urllib.parse # <-- ADDED for safe password handling
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
import kagglehub

# --- 1. CONFIGURATION & URI ASSEMBLY ---

from pymongo import MongoClient # import mongo client to connect
import json # import json to load credentials
import urllib.parse

# load credentials from json file
with open('credentials_mongodb.json') as f:
    login = json.load(f)

# assign credentials to variables
username = login['username']
password = urllib.parse.quote(login['password'])
host = login['host']
url = "mongodb+srv://{}:{}@{}/?retryWrites=true&w=majority".format(username, password, host)


DB_NAME = "mental_health_db"
COLLECTION_NAME = "support_data"
KAGGLE_DATASET = "jiscecseaiml/mental-health-dataset"
VECTOR_INDEX_NAME = "vector_index_mental_health"
EMBEDDING_MODEL = "models/gemini-embedding-001" # Dimensions: 768

# --- 2. DATA LOADING & PROCESSING ---

def load_and_chunk_data():
    """Loads a JSON dataset (like KB.json), converts it to a DataFrame, and chunks text."""
    data_path = 'C:\\Projects\\mental-health-chatbot\\KB.json'
    print(f"1. Loading dataset from: {data_path}")

    if not os.path.exists(data_path):
        print("Error: Dataset file not found.")
        return []

    # --- Load JSON ---
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    tags, patterns, responses = [], [], []

    for intent in data.get('intents', []):
        for pattern in intent.get('patterns', []):
            tags.append(intent.get('tag', 'unknown'))
            patterns.append(pattern)
            responses.append(intent.get('responses', [None])[0])

    # --- Convert to DataFrame for inspection (optional) ---
    df = pd.DataFrame({'tag': tags, 'pattern': patterns, 'response': responses})
    print(f"Loaded {len(df)} entries from JSON.")

    # --- Convert to LangChain Documents ---
    documents = []
    for _, row in df.iterrows():
        # Combine pattern + response for richer context
        content = f"User: {row['pattern']}\nBot: {row['response']}"
        metadata = {"tag": row['tag'], "source_dataset": "mental-health-dataset"}
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Created {len(documents)} documents.")

    # --- Split text into chunks for embedding ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"2. Split into {len(chunks)} chunks for embedding.")
    return chunks

# --- 3. MONGODB INGESTION ---

def ingest_to_mongodb(chunks):
    """Initializes embeddings and uploads chunks to MongoDB Atlas Vector Search."""
    print("4. Initializing MongoDB and Embeddings...")
    
    # Connect to MongoDB
    client = MongoClient(url) # Use the newly assembled URI
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Clear previous data (optional, but good practice for fresh ingestion)
    # print(f"Clearing existing data from {DB_NAME}.{COLLECTION_NAME}...")
    # collection.delete_many({})

    # Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)

    # Create the Vector Store and upload documents
    print("5. Generating embeddings and uploading to MongoDB...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
        index_name=VECTOR_INDEX_NAME,
        # text_key='text', # Default is 'text' which is fine
        embedding_key='embedding', # Default is 'embedding' which is fine
    )
    
    print("✅ Ingestion complete! Data is now ready for vector search.")
    print(f"Target Collection: {DB_NAME}.{COLLECTION_NAME}")
    print(f"Vector Index Used: {VECTOR_INDEX_NAME}")

if __name__ == "__main__":
    data_chunks = load_and_chunk_data()
    if data_chunks:
        ingest_to_mongodb(data_chunks)
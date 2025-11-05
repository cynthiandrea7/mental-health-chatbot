import os
import pandas as pd
import glob
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
import kagglehub

# --- 1. CONFIGURATION ---
MONGO_URI = os.environ.get('MONGO_URI')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not all([MONGO_URI, GOOGLE_API_KEY]):
    print("FATAL ERROR: MONGO_URI and GOOGLE_API_KEY must be set in environment variables.")
    exit()

DB_NAME = "mental_health_db"
COLLECTION_NAME = "support_data"
KAGGLE_DATASET = "jiscecseaiml/mental-health-dataset"
VECTOR_INDEX_NAME = "vector_index_mental_health"
EMBEDDING_MODEL = "models/gemini-embedding-001" # Dimensions: 768

# --- 2. DATA LOADING & PROCESSING ---

def load_and_chunk_data():
    """Downloads Kaggle data, loads the CSV, and chunks the text."""
    print(f"1. Downloading dataset: {KAGGLE_DATASET}...")
    try:
        # Downloads to a local cache directory
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"Dataset downloaded to: {path}")

    except Exception as e:
        print(f"Error downloading Kaggle dataset. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set.")
        print(f"Details: {e}")
        return []

    # Find the primary CSV file in the downloaded path (assuming it's a CSV)
    csv_files = glob.glob(os.path.join(path, '**/*.csv'), recursive=True)
    if not csv_files:
        print("Error: No CSV file found in the downloaded dataset.")
        return []
        
    data_file = csv_files[0]
    print(f"2. Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    # --- Data Cleaning and Preparation (Adjust based on dataset structure) ---
    # Assuming the dataset contains text columns relevant to the topic.
    # The search results suggest a dataset with 'Text' or 'Comment' columns. 
    # We will combine multiple text fields into one 'page_content'.
    
    # Identify key text columns (adjust these column names if your CSV is different)
    text_columns = ['text', 'Comment', 'AnswerText']
    
    # Find the first column that exists in the DataFrame
    content_column = next((col for col in text_columns if col in df.columns), None)

    if not content_column:
        print(f"Error: Could not find a suitable content column in the CSV. Tried: {text_columns}")
        return []

    print(f"Using column '{content_column}' for RAG content.")
    
    # Create LangChain Documents
    documents = []
    for _, row in df.iterrows():
        # Clean up text content
        content = str(row[content_column]).strip()
        
        # Optionally, include metadata (like survey ID or label if available)
        metadata = {
            "source_dataset": KAGGLE_DATASET,
        }
        
        if content and content != 'nan':
             documents.append(Document(page_content=content, metadata=metadata))

    print(f"Loaded {len(documents)} documents.")

    # Split documents into smaller chunks for better RAG performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"3. Split into {len(chunks)} chunks for embedding.")
    return chunks

# --- 3. MONGODB INGESTION ---

def ingest_to_mongodb(chunks):
    """Initializes embeddings and uploads chunks to MongoDB Atlas Vector Search."""
    print("4. Initializing MongoDB and Embeddings...")
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
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
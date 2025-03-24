import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# API Keys (use environment variables for security)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

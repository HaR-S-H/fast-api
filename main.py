import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

# FastAPI app
app = FastAPI()

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Google Gemini 2.0 Flash setup
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.3, "max_output_tokens": 1024})

# API Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# Root route
@app.get("/")
async def root():
    return {"message": "Bhagavad Gita API is running!"}

# Function to get similar verses from Pinecone
def query_verse(query: str, k: int):
    query_embedding = embedding_model.encode([query])[0]
    results = index.query(vector=query_embedding.tolist(), top_k=k, include_metadata=True)
    return [{"verse_text": match['metadata']['combined_text'], "similarity_score": match['score']} for match in results['matches']]

# RAG pipeline
def pipeline(query: str, k: int):
    relevant_documents = query_verse(query, k)
    context = "\n".join([doc['verse_text'] for doc in relevant_documents])

    system_message = """
    You are an expert in the Bhagavad Gita.
    You ONLY answer questions related to the Bhagavad Gita.
    If the question is unrelated, respond with: "I don't know. Not enough information received."
    """

    user_message = f"""
    Context:
    {context}
    ---------------------
    Answer the question: {query}
    ---------------------
    """

    chat = llm_model.start_chat(history=[{"role": "user", "parts": [system_message]}])
    response = chat.send_message(user_message)
    return response.text

# API Endpoint for querying the model
@app.post("/query")
def get_gita_response(request: QueryRequest):
    response = pipeline(request.query, request.top_k)
    return {"response": response}

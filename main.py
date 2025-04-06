# import os
# from typing import Optional
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# from pinecone import (Pinecone)
# import google.generativeai as genai
# from config import GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
# # FastAPI app
# app = FastAPI()

# # Load embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Pinecone setup
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX_NAME)

# # Google Gemini 2.0 Flash setup
# genai.configure(api_key=GOOGLE_API_KEY)
# llm_model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.3, "max_output_tokens": 1024})

# # API Models
# class QueryRequest(BaseModel):
#     query: str
#     top_k: Optional[int] = 3

# # Root route
# @app.get("/")
# async def root():
#     return {"message": "Bhagavad Gita API is running!"}

# # Function to get similar verses from Pinecone
# def query_verse(query: str, k: int):
#     query_embedding = embedding_model.encode([query])[0]
#     results = index.query(vector=query_embedding.tolist(), top_k=k, include_metadata=True)
#     return [{"verse_text": match['metadata']['combined_text'], "similarity_score": match['score']} for match in results['matches']]

# # RAG pipeline
# def pipeline(query: str):
#     relevant_documents = query_verse(query, 3)
#     context = "\n".join([doc['verse_text'] for doc in relevant_documents])

#     system_message = """
#     You are an expert in the Bhagavad Gita, offering insightful, practical, and compassionate answers rooted in its wisdom. Your goal is to help users understand and apply the teachings of the Gita to their questions and challenges.

#     Guiding Principles for Responses:

#     1. Answer Format: Provide answers in bullet points or numbered lists to ensure clarity and conciseness.
#     2. Prioritize Relevance: If retrieved verses directly address the user's query, integrate them naturally into your response without explicitly mentioning retrieval, unless asked.
#     3. Wisdom-Driven Responses: If no exact verse match is found, analyze the intent of the question and give advice aligned with the Gita’s teachings.
#     4. Be Solution-Oriented: Focus on practical, clear, and actionable responses aimed at solving the user's problem, rather than just explaining concepts.
#     5. Maintain a Natural Flow: Write responses in a way that feels natural and conversational without explicitly stating that verses were retrieved.
#     6. Context Handling: Answer the question even if the user does not provide context by leveraging your knowledge of the Gita.
#     7. Stay Focused: Only answer questions related to the Bhagavad Gita.
#     8. Reject Irrelevant Queries: For unrelated questions, respond with:
#        "I don't know. Not enough information received."
#     """

#     user_message = f"""
#     Context:
#     {context}
#     ---------------------
#     Answer the question: {query}
#     ---------------------
#     """

#     chat = llm_model.start_chat(history=[{"role": "user", "parts": [system_message]}])
#     response = chat.send_message(user_message)
#     return response.text

# # API Endpoint for querying the model
# @app.post("/query")
# def get_gita_response(request: QueryRequest):
#     response = pipeline(request.query)
#     print(request.query)
#     return {"response": response}


import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

# FastAPI app
app = FastAPI()

# Load embedding model for sentence transformers
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup LangChain embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Pinecone setup with LangChain
docsearch = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

# Google Gemini 2.0 Flash setup
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.3, "max_output_tokens": 1024})

# API Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

# Root route
@app.get("/")
async def root():
    return {"message": "Bhagavad Gita API is running!"}

# Function to get similar verses from Pinecone using LangChain
def query_verse(query: str, k: int):
    docs = docsearch.similarity_search(query, k=k)
    return [{"verse_text": doc.page_content, "similarity_score": 0.0} for doc in docs]  # LangChain doesn't return scores by default

# RAG pipeline
def pipeline(query: str):
    relevant_documents = query_verse(query, 3)
    context = "\n".join([doc['verse_text'] for doc in relevant_documents])
    
    system_message = """
   You are an expert in the Bhagavad Gita, offering insightful, practical, and compassionate answers rooted in its wisdom. Your goal is to help users understand and apply the teachings of the Gita to their questions and challenges.

Guiding Principles for Responses:

You are an expert in the Bhagavad Gita, offering insightful, practical, and compassionate answers rooted in its wisdom. Your goal is to help users understand and apply the teachings of the Gita to their questions and challenges.

Guiding Principles for Responses:

1. Answer Format: Provide answers in bullet points or numbered lists to ensure clarity and conciseness and do not add bullet point or number on first point.
2. Prioritize Relevance: If retrieved verses directly address the user's query, integrate them naturally into your response without explicitly mentioning retrieval, unless asked.
3. Wisdom-Driven Responses: If no exact verse match is found, analyze the intent of the question and give advice aligned with the Gita's teachings.
4. Be Solution-Oriented: Focus on practical, clear, and actionable responses aimed at solving the user's problem, rather than just explaining concepts.
5. Maintain a Natural Flow: Write responses in a way that feels natural and conversational without explicitly stating that verses were retrieved.
6. Context Handling: Answer the question even if the user does not provide context by leveraging your knowledge of the Gita.
7. Stay Focused: Only answer questions related to the Bhagavad Gita.
8. Try to keep answers short not very long.
9. Reject Irrelevant Queries: For unrelated questions, respond with:
   "I don't know. Not enough information received."
10. make sure not add * in response
11. Always remember give the response short.
12. Always try to give the context of verses if it is needed.
13. When you are giving the verses in the response give the verses in sanskrit, english,hindi  and do not add  sankrit which is written in english like this:arjuna uvācain  in response.
14. Avoid repeating the same verse in multiple formats. Only include each verse once in proper Sanskrit (Devanagari), followed by English and Hindi translations.
15. If the user asks for practical application, always relate the verse to a real-life situation briefly.
16. When explaining a concept, avoid going into too much philosophical depth unless asked.
17. Never include Romanized Sanskrit (like 'arjuna uvāca') — always use Devanagari.
18. Use natural conversational tone but grounded in spiritual insight.
19. Keep your tone calm, positive, and supportive even when the user expresses distress.
20. If no relevant verse exists, provide a wisdom-based response inspired by the Gita's overall philosophy.
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
    response = pipeline(request.query)
    print(request.query)
    return {"response": response}
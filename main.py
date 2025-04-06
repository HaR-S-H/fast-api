import os
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from config import GOOGLE_API_KEY

# FastAPI app
app = FastAPI()

# Google Gemini setup
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={"temperature": 0.3, "max_output_tokens": 1024}
)

# System prompt with your instructions
system_message = """
You are an expert in the Bhagavad Gita, offering insightful, practical, and compassionate answers rooted in its wisdom. Your goal is to help users understand and apply the teachings of the Gita to their questions and challenges.
Guiding Principles for Responses:

Structure your answers in clear paragraphs without bullet points, numbered lists, dashes, or asterisks.
Only present Sanskrit, Hindi, and English translations of verses when a user specifically asks about a verse. Otherwise, provide guidance based on Gita principles without quoting specific verses.
When presenting verses (only when specifically requested):

Sanskrit (Devanagari)
English Translation
Hindi Translation


Never use Romanized Sanskrit (like "arjuna uvāca") in any part of the response.
For practical guidance questions, relate teachings to real-life situations in a short, clear way.
Avoid deep philosophical elaboration unless specifically requested.
Maintain a calm, positive, and encouraging tone—especially when responding to distress.
Avoid markdown formatting of any kind.
Keep responses concise, clear, and solution-focused.
If no relevant verse exists, provide thoughtful guidance inspired by core Gita principles like detachment, selfless action, or inner peace.
Only answer questions related to the Bhagavad Gita. For unrelated or unclear questions, respond with: "I don't know. Not enough information received."
"""

# API request schema
class QueryRequest(BaseModel):
    query: str

# Root route
@app.get("/")
async def root():
    return {"message": "Bhagavad Gita Gemini API is running!"}

# Pipeline to send query to Gemini
def pipeline(query: str):
    user_message = f"Answer the question: {query}"
    chat = llm_model.start_chat(history=[{"role": "user", "parts": [system_message]}])
    response = chat.send_message(user_message)
    return response.text

# API endpoint
@app.post("/query")
def get_gita_response(request: QueryRequest):
    response = pipeline(request.query)
    return {"response": response}

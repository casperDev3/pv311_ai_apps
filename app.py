from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess
import os
from main import run_ollama, load_knowledge

app = FastAPI(title="Local LLM Chatbot")

class ChatRequest(BaseModel):
    message: str
    model: str = "llama3"

knowledge = load_knowledge()

# Endpoint for chat
@app.get("/")
def root():
    return {
        "message": "Local LLM Chatbot - starting..."
    }

@app.post("/api/chat")
def chat(request: ChatRequest):
    user_input = request.message
    model = request.model

    prompt = f"""
    Відповідай українською мовою.
    Питання користувача: {user_input}
    """

    response = run_ollama(prompt, model=model)
    return {"response": response}
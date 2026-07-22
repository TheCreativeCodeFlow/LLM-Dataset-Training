#!/usr/bin/env python3
"""
scripts/serve.py

Production FastAPI server for DSA Tutor.
Provides chat, hint, review, debug, complexity, and health endpoints using
a single shared TutorEngine singleton model instance.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from scripts.tutor_engine import TutorEngine, SYSTEM_PROMPTS

# Global tutor engine instance
tutor_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tutor_engine
    print("[Server] Initializing TutorEngine lifespan startup...")
    tutor_engine = TutorEngine()
    yield
    print("[Server] Lifespan shutdown completed.")

app = FastAPI(
    title="DSA Tutor API",
    version="1.2.0",
    description="Production-grade tutoring and inference API for Data Structures and Algorithms.",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_value: str = Depends(api_key_header)):
    expected_key = os.getenv("DSA_TUTOR_API_KEY", "dsa_tutor_prod_secure_key_2026")
    if not header_value or header_value != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return header_value

class ChatRequest(BaseModel):
    session_id: str = "default_session"
    query: str
    max_tokens: int = 256
    temperature: float = 0.3

class ChatResponse(BaseModel):
    session_id: str
    tutor_mode: str
    topic: str
    response: str

def get_final_response(generator) -> str:
    response_text = ""
    for chunk in generator:
        if chunk["token"] == "[DONE]":
            response_text = chunk["full_response"]
            break
    return response_text

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
async def chat(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded or initialized.")
        
    try:
        generator = tutor_engine.generate_response_stream(
            session_id=request.session_id,
            query=request.query
        )
        response_text = get_final_response(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode=session.tutor_mode,
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating tutor response: {str(e)}")

@app.post("/hint", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
async def hint(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response_stream(
            session_id=request.session_id,
            query=request.query,
            force_mode="hint_generator"
        )
        response_text = get_final_response(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode="hint_generator",
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating hint: {str(e)}")

@app.post("/review", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
async def review(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response_stream(
            session_id=request.session_id,
            query=request.query,
            force_mode="code_reviewer"
        )
        response_text = get_final_response(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode="code_reviewer",
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating review: {str(e)}")

@app.post("/debug", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
async def debug(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response_stream(
            session_id=request.session_id,
            query=request.query,
            force_mode="debugging_mentor"
        )
        response_text = get_final_response(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode="debugging_mentor",
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating debugging guide: {str(e)}")

@app.post("/complexity", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
async def complexity(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response_stream(
            session_id=request.session_id,
            query=request.query,
            force_mode="complexity_analyst"
        )
        response_text = get_final_response(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode="complexity_analyst",
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error analyzing complexity: {str(e)}")

@app.get("/health")
async def health():
    if tutor_engine is None or tutor_engine.loader.model is None:
        return {"status": "loading", "tutor_engine_loaded": False}
        
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    
    loader = tutor_engine.loader
    return {
        "status": "ok",
        "tutor_engine_loaded": True,
        "base_model": loader.base_model_name,
        "adapter": loader.adapter_path,
        "tokenizer": loader.base_model_name,
        "device": loader.device,
        "memory_usage_mb": round(mem_mb, 2),
        "load_status": "complete",
        "model_hash": loader.model_hash,
        "adapter_hash": loader.adapter_hash,
        "available_modes": list(SYSTEM_PROMPTS.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
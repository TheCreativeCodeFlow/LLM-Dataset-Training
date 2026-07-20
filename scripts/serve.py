#!/usr/bin/env python3
"""
scripts/serve.py

FastAPI inference server for DSA Tutor Platform.
Exposes tutor mode endpoints, manages session routing, memory, and safety filters.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
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
    version="1.1.0",
    description="Production-grade tutoring and inference API for Data Structures and Algorithms.",
    lifespan=lifespan
)

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded or initialized.")
        
    try:
        # Run prompt routing & response pipeline
        generator = tutor_engine.generate_response(
            session_id=request.session_id,
            query=request.query
        )
        response_text = next(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode=session.tutor_mode,
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating tutor response: {str(e)}")

@app.post("/hint", response_model=ChatResponse)
async def hint(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response(
            session_id=request.session_id,
            query=request.query,
            force_mode="hint_generator"
        )
        response_text = next(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode="hint_generator",
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating hint: {str(e)}")

@app.post("/review", response_model=ChatResponse)
async def review(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response(
            session_id=request.session_id,
            query=request.query,
            force_mode="code_reviewer"
        )
        response_text = next(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode="code_reviewer",
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating review: {str(e)}")

@app.post("/debug", response_model=ChatResponse)
async def debug(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response(
            session_id=request.session_id,
            query=request.query,
            force_mode="debugging_mentor"
        )
        response_text = next(generator)
        session = tutor_engine.get_or_create_session(request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            tutor_mode="debugging_mentor",
            topic=session.current_topic,
            response=response_text
        )
    except Exception as e:
        raise HTTPException(500, f"Error generating debugging guide: {str(e)}")

@app.post("/complexity", response_model=ChatResponse)
async def complexity(request: ChatRequest):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded.")
    try:
        generator = tutor_engine.generate_response(
            session_id=request.session_id,
            query=request.query,
            force_mode="complexity_analyst"
        )
        response_text = next(generator)
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
    return {
        "status": "ok",
        "tutor_engine_loaded": tutor_engine is not None,
        "available_modes": list(SYSTEM_PROMPTS.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
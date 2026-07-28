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
from fastapi.responses import StreamingResponse
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

async def stream_tutor_endpoint(category: str, request: ChatRequest, force_mode: str = None):
    if tutor_engine is None:
        raise HTTPException(503, "TutorEngine not fully loaded or initialized.")
        
    def event_generator():
        try:
            generator = tutor_engine.generate_response_stream(
                session_id=request.session_id,
                query=request.query,
                force_mode=force_mode
            )
            
            header_sent = False
            mode_name = ""
            accumulated_raw = ""
            
            for chunk in generator:
                if chunk["token"] != "[DONE]":
                    token = chunk["token"]
                    accumulated_raw += token
                    if not header_sent:
                        mode = chunk["tutor_mode"]
                        mode_name = mode.replace('_', ' ').title()
                        header = f"**[{mode_name}]**\n\n### Concept & Explanation\n"
                        yield f"data: {header}\n\n"
                        header_sent = True
                    yield f"data: {token}\n\n"
                else:
                    final_response = chunk["full_response"]
                    header = f"**[{mode_name}]**\n\n### Concept & Explanation\n"
                    prefix_len = len(header) + len(accumulated_raw)
                    suffix = final_response[prefix_len:]
                    if suffix:
                        yield f"data: {suffix}\n\n"
                    yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ⚠️ Error generating response: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/chat", dependencies=[Depends(get_api_key)])
async def chat(request: ChatRequest):
    return await stream_tutor_endpoint("chat", request)

@app.post("/hint", dependencies=[Depends(get_api_key)])
async def hint(request: ChatRequest):
    return await stream_tutor_endpoint("hint", request, force_mode="hint_generator")

@app.post("/review", dependencies=[Depends(get_api_key)])
async def review(request: ChatRequest):
    return await stream_tutor_endpoint("review", request, force_mode="code_reviewer")

@app.post("/debug", dependencies=[Depends(get_api_key)])
async def debug(request: ChatRequest):
    return await stream_tutor_endpoint("debug", request, force_mode="debugging_mentor")

@app.post("/complexity", dependencies=[Depends(get_api_key)])
async def complexity(request: ChatRequest):
    return await stream_tutor_endpoint("complexity", request, force_mode="complexity_analyst")

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
        "available_modes": list(SYSTEM_PROMPTS.keys()),
        "retriever": "Hybrid Chunks Retriever",
        "embedding_model": "Pure Python TF-IDF Vectorizer",
        "vector_store": "Lightweight NumPy Chunks Store",
        "knowledge_base_size_docs": len(tutor_engine.vector_store.chunks)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
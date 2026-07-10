#!/usr/bin/env python3
"""FastAPI inference server for DSA Tutor."""

import yaml
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from contextlib import asynccontextmanager


model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    with open("configs/train_config.yaml") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        device_map="auto",
        torch_dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, train_cfg["output_dir"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    yield


app = FastAPI(title="DSA Tutor API", lifespan=lifespan)


class Query(BaseModel):
    problem: str
    language: str = "python"
    difficulty: str = "medium"
    topic: str = ""
    algorithm: str = ""
    max_tokens: int = 512
    temperature: float = 0.7


class Response(BaseModel):
    solution: str


@app.post("/solve", response_model=Response)
async def solve(query: Query):
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model not loaded")

    system_prompt = "You are an expert DSA tutor. Provide clear explanations and optimal solutions."
    tags = []
    if query.difficulty:
        tags.append(f"[difficulty: {query.difficulty}]")
    if query.topic:
        tags.append(f"[topic: {query.topic}]")
    if query.algorithm:
        tags.append(f"[algorithm: {query.algorithm}]")
    tag_str = " ".join(tags)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{query.problem} {tag_str}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=query.max_tokens,
            temperature=query.temperature,
            do_sample=query.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return Response(solution=response.strip())


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
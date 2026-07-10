import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn


class Query(BaseModel):
    problem: str
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95


class Response(BaseModel):
    solution: str


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


config = load_config("configs/infer_config.yaml")
model_config = config["model"]
infer_config = config["inference"]
server_config = config["server"]

tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_config["base_model"],
    torch_dtype=getattr(torch, model_config["torch_dtype"]),
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, model_config["adapter_path"])
model.eval()

system_prompt = config["prompt"]["system"]

app = FastAPI(title="DSA Tutor LLM", version="1.0.0")


@app.post("/solve", response_model=Response)
async def solve(query: Query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query.problem},
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
            top_p=query.top_p,
            do_sample=query.temperature > 0,
            repetition_penalty=infer_config["repetition_penalty"],
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return Response(solution=response.strip())


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_config["base_model"]}


if __name__ == "__main__":
    uvicorn.run(app, host=server_config["host"], port=server_config["port"], workers=server_config["workers"])
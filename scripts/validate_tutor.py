#!/usr/bin/env python3
"""
scripts/validate_tutor.py

Runs identical prompts through:
1. Base Model (No Adapter, No RAG)
2. PEFT Model (LoRA Adapter, No RAG)
3. PEFT + RAG Model (LoRA Adapter + RAG)
Judges responses and compiles comparative validation statistics.
"""

import os
import sys
import time
import json
import yaml
import torch

torch.set_num_threads(12)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.tutor_engine import TutorEngine
from scripts.eval_engine import EvalEngine

VALIDATION_PROMPTS = [
    {
        "query": "What is the time complexity of binary search?",
        "topic": "Binary Search",
        "mode": "complexity_analyst",
        "lang": "none"
    },
    {
        "query": "Give me a progressive hint on detecting a cycle in a linked list.",
        "topic": "Linked List",
        "mode": "hint_generator",
        "lang": "python"
    },
    {
        "query": "Review this Python code to remove duplicates:\n```python\ndef rm(arr):\n    for x in arr: arr.remove(x)\n```",
        "topic": "Arrays",
        "mode": "code_reviewer",
        "lang": "python"
    },
    {
        "query": "Why does my recursive graph DFS loop infinitely when searching paths?",
        "topic": "Graphs",
        "mode": "debugging_mentor",
        "lang": "cpp"
    },
    {
        "query": "Explain how AVL trees balance height after insertion.",
        "topic": "AVL",
        "mode": "beginner_tutor",
        "lang": "none"
    }
]

def clean_gen(generator):
    for chunk in generator:
        if chunk["token"] == "[DONE]":
            return chunk["full_response"], chunk["metrics"]
    return "", {}

def main():
    print("=== Commencing Validation Suite and Comparative baseline run ===")
    
    engine = TutorEngine()
    evaluator = EvalEngine()
    
    # Warmup
    list(engine.generate_response_stream("warmup", "Hello", use_rag=True))
    
    report_data = []
    
    for idx, case in enumerate(VALIDATION_PROMPTS):
        query = case["query"]
        mode = case["mode"]
        topic = case["topic"]
        lang = case["lang"]
        
        print(f"\n[{idx+1}/{len(VALIDATION_PROMPTS)}] Evaluating query: '{query[:50]}...'")
        
        # 1. Base Model Run
        print("  - Running Base Model...")
        base_res, base_m = clean_gen(engine.generate_response_stream(
            session_id=f"base_{idx}",
            query=query,
            force_mode=mode,
            use_rag=False,
            disable_adapter=True
        ))
        
        # 2. PEFT LoRA Run
        print("  - Running PEFT Model...")
        peft_res, peft_m = clean_gen(engine.generate_response_stream(
            session_id=f"peft_{idx}",
            query=query,
            force_mode=mode,
            use_rag=False,
            disable_adapter=False
        ))
        
        # 3. PEFT + RAG Run
        print("  - Running PEFT + RAG Model...")
        rag_res, rag_m = clean_gen(engine.generate_response_stream(
            session_id=f"rag_{idx}",
            query=query,
            force_mode=mode,
            use_rag=True,
            disable_adapter=False
        ))
        
        # Retrieve target complexities from knowledge base for validation
        kb_results = engine.vector_store.retrieve(query, top_k=1)
        dummy_context = ""
        if len(kb_results) > 0:
            doc = kb_results[0][0]
            dummy_context = doc['content']
            
        # Score each response
        base_score = evaluator.evaluate_response(query, dummy_context, base_res, mode, stage=2)
        peft_score = evaluator.evaluate_response(query, dummy_context, peft_res, mode, stage=2)
        rag_score = evaluator.evaluate_response(query, dummy_context, rag_res, mode, stage=2)
        
        report_data.append({
            "query": query,
            "topic": topic,
            "mode": mode,
            "language": lang,
            "base": {
                "response": base_res,
                "scores": base_score
            },
            "peft": {
                "response": peft_res,
                "scores": peft_score
            },
            "rag": {
                "response": rag_res,
                "scores": rag_score
            }
        })
        
    # Write logs/validation_results.json
    os.makedirs("logs", exist_ok=True)
    with open("logs/validation_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    print("\nValidation baseline comparisons run completed successfully!")

if __name__ == "__main__":
    # Temporarily set max_new_tokens: 30 in configs/inference.yaml to accelerate CPU evaluation
    with open("configs/inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    old_max = cfg["inference"]["max_new_tokens"]
    cfg["inference"]["max_new_tokens"] = 30
    with open("configs/inference.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
        
    try:
        main()
    finally:
        with open("configs/inference.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg["inference"]["max_new_tokens"] = old_max
        with open("configs/inference.yaml", "w") as f:
            yaml.safe_dump(cfg, f)
        print("[validate_tutor] Restored original inference config max_new_tokens.")

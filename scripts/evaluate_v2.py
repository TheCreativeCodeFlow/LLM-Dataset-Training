#!/usr/bin/env python3
"""
scripts/evaluate_v2.py

Comparative benchmark suite running v1 (parametric only) vs v2 (RAG-augmented)
on representative DSA prompts. Measures latency, complexity accuracy, and solution leakage.
"""

import os
import sys
import time
import json
import yaml
import torch

# Configure 12 threads for fast local CPU evaluation
torch.set_num_threads(12)

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.tutor_engine import TutorEngine

BENCHMARK_PROMPTS = [
    {
        "topic": "Binary Search",
        "mode": "complexity_analyst",
        "query": "What is the time and space complexity of binary search?"
    },
    {
        "topic": "Hash Maps",
        "mode": "beginner_tutor",
        "query": "Explain how hash maps map keys to values and handle collisions."
    },
    {
        "topic": "Linked List",
        "mode": "hint_generator",
        "query": "Give me a hint on how to detect a loop inside a linked list."
    },
    {
        "topic": "Dynamic Programming",
        "mode": "beginner_tutor",
        "query": "What is dynamic programming and when should I use memoization?"
    },
    {
        "topic": "Graphs",
        "mode": "debugging_mentor",
        "query": "Why does my graph depth first search loop infinitely? Can you guide me?"
    }
]

def clean_gen(generator):
    for chunk in generator:
        if chunk["token"] == "[DONE]":
            return chunk["full_response"], chunk["metrics"]
    return "", {}

def run_benchmark():
    print("=== Commencing Phase 5: v1 vs v2 Comparative Evaluation ===")
    
    engine = TutorEngine()
    results = []
    
    # 1. Warm up
    print("Warming up engine...")
    list(engine.generate_response_stream("warmup", "Hello", use_rag=True))
    
    v1_total_latency = 0.0
    v2_total_latency = 0.0
    v1_correct_complexities = 0
    v2_correct_complexities = 0
    v1_hint_leaks = 0
    v2_hint_leaks = 0
    
    for idx, item in enumerate(BENCHMARK_PROMPTS):
        topic = item["topic"]
        mode = item["mode"]
        query = item["query"]
        
        print(f"\n[{idx+1}/{len(BENCHMARK_PROMPTS)}] Evaluating query: '{query}' (Topic: {topic})")
        
        # --- Evaluate v1 (No RAG) ---
        print("  - Running in v1 Mode (Parametric)...")
        start_v1 = time.time()
        v1_text, v1_metrics = clean_gen(engine.generate_response_stream(
            session_id=f"v1_{idx}",
            query=query,
            force_mode=mode,
            use_rag=False
        ))
        v1_lat = time.time() - start_v1
        v1_total_latency += v1_lat
        
        # --- Evaluate v2 (RAG) ---
        print("  - Running in v2 Mode (RAG-Augmented)...")
        start_v2 = time.time()
        v2_text, v2_metrics = clean_gen(engine.generate_response_stream(
            session_id=f"v2_{idx}",
            query=query,
            force_mode=mode,
            use_rag=True
        ))
        v2_lat = time.time() - start_v2
        v2_total_latency += v2_lat
        
        # Retrieve target complexities from knowledge base for validation
        kb_results = engine.vector_store.retrieve(query, top_k=1)
        expected_time = "N/A"
        expected_space = "N/A"
        if len(kb_results) > 0:
            doc = kb_results[0][0]
            expected_time = doc["complexities"].get("time", doc["complexities"].get("search", "N/A")).lower()
            expected_space = doc["complexities"].get("space", "N/A").lower()
            
        # Check complexity correctness
        v1_comp_ok = expected_time in v1_text.lower() or expected_space in v1_text.lower()
        v2_comp_ok = expected_time in v2_text.lower() or expected_space in v2_text.lower()
        
        if v1_comp_ok:
            v1_correct_complexities += 1
        if v2_comp_ok:
            v2_correct_complexities += 1
            
        # Check hint leakage (for hint_generator mode, check if code blocks exist)
        v1_leak = (mode == "hint_generator" and "```" in v1_text)
        v2_leak = (mode == "hint_generator" and "```" in v2_text)
        
        if v1_leak:
            v1_hint_leaks += 1
        if v2_leak:
            v2_hint_leaks += 1
            
        results.append({
            "query": query,
            "topic": topic,
            "mode": mode,
            "v1": {
                "response": v1_text,
                "latency_seconds": v1_lat,
                "complexity_matched": v1_comp_ok,
                "solution_leaked": v1_leak
            },
            "v2": {
                "response": v2_text,
                "latency_seconds": v2_lat,
                "complexity_matched": v2_comp_ok,
                "solution_leaked": v2_leak,
                "retrieval_latency": v2_metrics.get("retrieval_latency_seconds", 0.0)
            }
        })
        
    num_prompts = len(BENCHMARK_PROMPTS)
    v1_avg_lat = v1_total_latency / num_prompts
    v2_avg_lat = v2_total_latency / num_prompts
    v1_comp_acc = (v1_correct_complexities / num_prompts) * 100
    v2_comp_acc = (v2_correct_complexities / num_prompts) * 100
    
    summary = {
        "v1_average_latency": v1_avg_lat,
        "v2_average_latency": v2_avg_lat,
        "v1_complexity_accuracy_pct": v1_comp_acc,
        "v2_complexity_accuracy_pct": v2_comp_acc,
        "v1_total_hint_leaks": v1_hint_leaks,
        "v2_total_hint_leaks": v2_hint_leaks
    }
    
    print("\n=====================================")
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=====================================")
    print(f"v1 (Parametric Only) Average Latency: {v1_avg_lat:.2f}s")
    print(f"v2 (RAG-Augmented) Average Latency:   {v2_avg_lat:.2f}s")
    print(f"v1 Complexity Extraction Accuracy:    {v1_comp_acc:.1f}%")
    print(f"v2 Complexity Extraction Accuracy:    {v2_comp_acc:.1f}%")
    print(f"v1 Solution Leaks (Hint Mode):        {v1_hint_leaks}")
    print(f"v2 Solution Leaks (Hint Mode):        {v2_hint_leaks}")
    
    # Save report
    os.makedirs("logs", exist_ok=True)
    report_path = "logs/tutor_v2_comparison.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nReport successfully saved to: {report_path}")

if __name__ == "__main__":
    # Temporarily set max_new_tokens: 30 in configs/inference.yaml to accelerate CPU evaluation
    with open("configs/inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    old_max = cfg["inference"]["max_new_tokens"]
    cfg["inference"]["max_new_tokens"] = 30
    with open("configs/inference.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
        
    try:
        run_benchmark()
    finally:
        # Restore original max_new_tokens configuration
        with open("configs/inference.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg["inference"]["max_new_tokens"] = old_max
        with open("configs/inference.yaml", "w") as f:
            yaml.safe_dump(cfg, f)
        print("[evaluate_v2] Restored original inference config max_new_tokens.")

#!/usr/bin/env python3
"""
scripts/benchmark_base_model.py

Evaluates the base model (or fine-tuned adapter) on a fixed benchmark set.
Measures latency, GPU memory, tokens, and educational quality.
"""

import os
import sys
import json
import time
import argparse
import re
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Bypassing large integer string conversion limits
sys.set_int_max_str_digits(0)


def load_config(path: str) -> dict:
    """Helper to load train configuration if available."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def generate_benchmark_file(filepath: str):
    """Generates the balanced 100-conversation dsa_benchmark.json dataset programmatically."""
    system_prompt = "You are an expert DSA tutor. Provide clear, step-by-step explanations, hints, and code reviews."
    
    prompts_db = {
        ("Arrays", "concept explanation"): [
            "Explain how 2D arrays are stored in memory.",
            "Explain the concept of prefix sums in arrays."
        ],
        ("Arrays", "hint generation"): [
            "I'm trying to find duplicate elements in an array. Can you give me a hint?",
            "I need to rotate an array. What's a good hint?"
        ],
        ("Arrays", "bug diagnosis"): [
            "My array index is out of bounds in this loop. Why does it happen?",
            "My array is not reversing properly in-place. What is the bug?"
        ],
        ("Arrays", "complexity analysis"): [
            "What is the time complexity of sorting an array of size N?",
            "What is the space complexity of copying an array?"
        ],
        ("Arrays", "interview follow-up"): [
            "Can you optimize array rotation to O(1) extra space?",
            "How would you find the maximum subarray sum in O(N) time?"
        ],
        # Strings
        ("Strings", "concept explanation"): [
            "Explain the difference between mutable and immutable strings.",
            "How does a Trie structure represent strings?"
        ],
        ("Strings", "hint generation"): [
            "I need to check if a string is a palindrome. Can you give me a hint?",
            "I need to find the first non-repeating character. Give me a hint."
        ],
        ("Strings", "bug diagnosis"): [
            "My string concatenation is very slow in Python. Why?",
            "My string slicing has an off-by-one index error. Can you find the bug?"
        ],
        ("Strings", "complexity analysis"): [
            "What is the time complexity of comparing two strings of length N?",
            "What is the space complexity of a suffix tree?"
        ],
        ("Strings", "interview follow-up"): [
            "Can we find the longest palindromic substring in O(N^2)?",
            "What if the input strings are too large to fit in memory?"
        ],
        # Hash Maps
        ("Hash Maps", "concept explanation"): [
            "Explain how hash collisions are resolved.",
            "What is the load factor of a hash table?"
        ],
        ("Hash Maps", "hint generation"): [
            "I'm trying to find two numbers that add up to a target. Can you give me a hint?",
            "Give me a hint to detect duplicate elements in a stream."
        ],
        ("Hash Maps", "bug diagnosis"): [
            "My hashmap lookup is returning None even though the key should exist. Why?",
            "Why is my custom object key causing slow lookups in python dictionary?"
        ],
        ("Hash Maps", "complexity analysis"): [
            "What is the amortized complexity of hashmap operations?",
            "What is the worst-case space complexity of a hashmap?"
        ],
        ("Hash Maps", "interview follow-up"): [
            "How would you design a thread-safe hash map?",
            "What if hash collisions make operations O(N)?"
        ],
        # Linked Lists
        ("Linked Lists", "concept explanation"): [
            "Explain the difference between a singly linked list and a doubly linked list.",
            "Explain how to detect loops in a linked list."
        ],
        ("Linked Lists", "hint generation"): [
            "I need to reverse a linked list. Give me a hint.",
            "How do I find the middle of a linked list in one pass? Give me a hint."
        ],
        ("Linked Lists", "bug diagnosis"): [
            "My loop detection code is stuck in an infinite loop. Why?",
            "My linked list reverse method loses the tail node. What is the bug?"
        ],
        ("Linked Lists", "complexity analysis"): [
            "What is the complexity of inserting a node at the head vs tail?",
            "What is the space complexity of recursive linked list traversal?"
        ],
        ("Linked Lists", "interview follow-up"): [
            "Can you reverse a linked list in O(1) auxiliary space?",
            "How would you merge two sorted linked lists without extra memory?"
        ],
        # Trees
        ("Trees", "concept explanation"): [
            "Explain the properties of a Binary Search Tree (BST).",
            "What is the difference between BFS and DFS tree traversal?"
        ],
        ("Trees", "hint generation"): [
            "I want to find the lowest common ancestor of two nodes in a BST. Give me a hint.",
            "I need to check if a binary tree is balanced. Give me a hint."
        ],
        ("Trees", "bug diagnosis"): [
            "My post-order traversal function is printing nodes in the wrong order. Why?",
            "My tree height function causes a stack overflow. What is the bug?"
        ],
        ("Trees", "complexity analysis"): [
            "What is the worst-case search complexity in a BST?",
            "What is the space complexity of tree BFS traversal?"
        ],
        ("Trees", "interview follow-up"): [
            "Can we implement tree traversal without recursion or call stack?",
            "How would you serialize and deserialize a binary tree?"
        ],
        # Graphs
        ("Graphs", "concept explanation"): [
            "Explain the difference between an adjacency list and an adjacency matrix.",
            "What is topological sorting in a DAG?"
        ],
        ("Graphs", "hint generation"): [
            "I want to find the shortest path in a graph. Give me a hint.",
            "How do I detect a cycle in an undirected graph? Give me a hint."
        ],
        ("Graphs", "bug diagnosis"): [
            "My DFS traversal is visiting the same node multiple times. Why?",
            "My Dijkstra's implementation is not finding the shortest path on negative weights. Why?"
        ],
        ("Graphs", "complexity analysis"): [
            "What is the complexity of BFS on a graph with V vertices and E edges?",
            "What is the space complexity of DFS call stack?"
        ],
        ("Graphs", "interview follow-up"): [
            "How would you handle dynamic edge updates in a shortest path graph?",
            "What if the graph vertices cannot fit on a single server?"
        ],
        # Binary Search
        ("Binary Search", "concept explanation"): [
            "Explain how binary search splits the search space.",
            "When can binary search be applied on non-sorted inputs?"
        ],
        ("Binary Search", "hint generation"): [
            "I need to find the square root of an integer. Give me a hint.",
            "I need to find the peak element in an unsorted array. Give me a hint."
        ],
        ("Binary Search", "bug diagnosis"): [
            "My binary search is stuck in an infinite loop. Why?",
            "My index midpoint calculation is causing integer overflow. How do I fix it?"
        ],
        ("Binary Search", "complexity analysis"): [
            "What is the complexity of binary search?",
            "What is the space complexity of recursive binary search?"
        ],
        ("Binary Search", "interview follow-up"): [
            "Can we apply binary search on a linked list?",
            "How would you find a target in a rotated sorted array in O(log N) time?"
        ],
        # Sliding Window
        ("Sliding Window", "concept explanation"): [
            "Explain the difference between fixed-size and variable-size sliding windows.",
            "What is the sliding window pattern?"
        ],
        ("Sliding Window", "hint generation"): [
            "I need to find the longest substring without repeating characters. Give me a hint.",
            "I want to find the maximum sum subarray of size K. Give me a hint."
        ],
        ("Sliding Window", "bug diagnosis"): [
            "My window right pointer goes out of bounds. Why?",
            "My sliding window minimum length is incorrect on small inputs. What is the bug?"
        ],
        ("Sliding Window", "complexity analysis"): [
            "What is the time complexity of the sliding window when nested loops are present?",
            "What is the space complexity of a sliding window using a set?"
        ],
        ("Sliding Window", "interview follow-up"): [
            "How would you handle streaming data where window size grows dynamically?",
            "Can you solve the sliding window maximum problem in O(N) time?"
        ],
        # Dynamic Programming
        ("Dynamic Programming", "concept explanation"): [
            "Explain the difference between Memoization and Tabulation.",
            "What are the two key properties of dynamic programming?"
        ],
        ("Dynamic Programming", "hint generation"): [
            "I need to solve the coin change problem. Give me a hint.",
            "I want to find the longest common subsequence of two strings. Give me a hint."
        ],
        ("Dynamic Programming", "bug diagnosis"): [
            "My DP array indexing is causing negative bounds error. Why?",
            "My memoization key is too slow to hash. What is the bug?"
        ],
        ("Dynamic Programming", "complexity analysis"): [
            "What is the complexity of the 0/1 knapsack problem?",
            "What is the space complexity of the Fibonacci tabulation method?"
        ],
        ("Dynamic Programming", "interview follow-up"): [
            "Can we optimize the space complexity of the knapsack DP table to O(W)?",
            "How would you solve the edit distance problem in O(N) space?"
        ],
        # Greedy
        ("Greedy", "concept explanation"): [
            "Explain the greedy choice property.",
            "Why doesn't greedy always yield the global optimum?"
        ],
        ("Greedy", "hint generation"): [
            "I need to solve the interval scheduling problem. Give me a hint.",
            "I want to find the minimum number of coins to make change. Give me a hint."
        ],
        ("Greedy", "bug diagnosis"): [
            "My greedy algorithm fails on certain denominations. Why?",
            "My sorting step in greedy scheduling has an incorrect comparator. What is the bug?"
        ],
        ("Greedy", "complexity analysis"): [
            "What is the complexity of Huffman coding?",
            "What is the complexity of Kruskal's MST algorithm?"
        ],
        ("Greedy", "interview follow-up"): [
            "Can you prove why the greedy choice works for interval scheduling?",
            "How would you handle fraction values in fractional knapsack?"
        ]
    }
    
    benchmark_data = []
    idx = 0
    for (topic, type_name), variations in prompts_db.items():
        for var_idx, user_prompt in enumerate(variations):
            benchmark_data.append({
                "id": f"bench_{idx}",
                "topic": topic,
                "difficulty": "medium" if var_idx == 0 else "hard",
                "conversation_type": type_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            })
            idx += 1
            
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Generated balanced benchmark with {len(benchmark_data)} conversations at: {filepath}")


def compute_educational_metrics(response: str, topic: str, convo_type: str) -> dict:
    """Pedagogical scoring heuristics based on keyword density, scaffolding, and Big-O mentions."""
    metrics = {
        "concept_correctness": 0,
        "hint_usefulness": 0,
        "explanation_completeness": 0,
        "complexity_correctness": 0,
        "pedagogical_progression": 0
    }
    
    keywords_by_topic = {
        "Arrays": ["array", "index", "element", "contiguous", "subarray"],
        "Strings": ["string", "char", "substring", "palindrome", "anagram", "trie"],
        "Hash Maps": ["hash", "key", "value", "collision", "map", "dictionary"],
        "Linked Lists": ["node", "pointer", "next", "head", "tail", "list"],
        "Trees": ["node", "bst", "root", "left", "right", "traversal", "child"],
        "Graphs": ["vertex", "edge", "adjacency", "bfs", "dfs", "cycle", "path"],
        "Binary Search": ["search", "mid", "half", "sorted", "divide", "bound"],
        "Sliding Window": ["window", "subarray", "pointer", "contiguous", "slide"],
        "Dynamic Programming": ["state", "subproblem", "memoization", "table", "tabulation"],
        "Greedy": ["greedy", "local", "optimum", "choice", "sort", "schedule"]
    }
    
    words = response.lower()
    topic_kws = keywords_by_topic.get(topic, [])
    match_count = sum(1 for kw in topic_kws if kw in words)
    metrics["concept_correctness"] = min(5, max(1, int((match_count / len(topic_kws)) * 5) if topic_kws else 3))
    
    if convo_type == "hint generation":
        if "```" in response or "def " in response:
            metrics["hint_usefulness"] = 1
        else:
            metrics["hint_usefulness"] = 4 if len(response) > 100 else 3
    else:
        metrics["hint_usefulness"] = 3
        
    steps = sum(1 for marker in ["1.", "2.", "first", "second", "step", "however", "therefore"] if marker in words)
    length_score = min(3, len(response) // 200)
    metrics["explanation_completeness"] = min(5, 1 + steps + length_score)
    
    if "o(" in words:
        metrics["complexity_correctness"] = 5
    elif "complexity" in words:
        metrics["complexity_correctness"] = 3
    else:
        metrics["complexity_correctness"] = 2
        
    scaffolds = ["try", "think", "what if", "let's", "how", "consider", "challenge"]
    scaffold_count = sum(1 for word in scaffolds if word in words)
    metrics["pedagogical_progression"] = min(5, 1 + scaffold_count)
    
    return metrics


def generate_reports(results: list, results_file: str, summary_file: str):
    """Saves raw outputs to JSON and builds a detailed performance markdown summary."""
    import numpy as np
    
    total = len(results)
    latencies = [r["latency"] for r in results]
    prompt_tokens = [r["prompt_tokens"] for r in results]
    resp_tokens = [r["response_tokens"] for r in results]
    
    avg_latency = np.mean(latencies) if latencies else 0
    p50_latency = np.percentile(latencies, 50) if latencies else 0
    p90_latency = np.percentile(latencies, 90) if latencies else 0
    
    avg_prompt = np.mean(prompt_tokens) if prompt_tokens else 0
    avg_resp = np.mean(resp_tokens) if resp_tokens else 0
    
    topic_stats = {}
    for r in results:
        t = r["topic"]
        if t not in topic_stats:
            topic_stats[t] = []
        topic_stats[t].append(r)
        
    type_stats = {}
    for r in results:
        ct = r["conversation_type"]
        if ct not in type_stats:
            type_stats[ct] = []
        type_stats[ct].append(r)
        
    # Write JSON results
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Write MD summary
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Summary Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- **Total Samples:** {total}\n")
        f.write(f"- **Average Latency:** {avg_latency:.2f} s\n")
        f.write(f"- **P50 Latency:** {p50_latency:.2f} s\n")
        f.write(f"- **P90 Latency:** {p90_latency:.2f} s\n")
        f.write(f"- **Average Prompt Tokens:** {avg_prompt:.1f}\n")
        f.write(f"- **Average Response Tokens:** {avg_resp:.1f}\n\n")
        
        f.write("## Per-Topic Statistics\n\n")
        f.write("| Topic | Count | Avg Latency (s) | Avg Resp Tokens | Concept Score | Complexity Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for t, items in topic_stats.items():
            avg_t_lat = np.mean([x["latency"] for x in items])
            avg_t_resp = np.mean([x["response_tokens"] for x in items])
            avg_t_concept = np.mean([x["educational_metrics"]["concept_correctness"] for x in items])
            avg_t_complex = np.mean([x["educational_metrics"]["complexity_correctness"] for x in items])
            f.write(f"| {t} | {len(items)} | {avg_t_lat:.2f} | {avg_t_resp:.1f} | {avg_t_concept:.2f} | {avg_t_complex:.2f} |\n")
            
        f.write("\n## Per-Conversation Type Statistics\n\n")
        f.write("| Conversation Type | Count | Avg Latency (s) | Avg Resp Tokens | Pedagogical Score |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for ct, items in type_stats.items():
            avg_ct_lat = np.mean([x["latency"] for x in items])
            avg_ct_resp = np.mean([x["response_tokens"] for x in items])
            avg_ct_ped = np.mean([x["educational_metrics"]["pedagogical_progression"] for x in items])
            f.write(f"| {ct} | {len(items)} | {avg_ct_lat:.2f} | {avg_ct_resp:.1f} | {avg_ct_ped:.2f} |\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline and Adapter Benchmarking Framework")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter to evaluate")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation cases")
    args = parser.parse_args()

    benchmark_path = "benchmarks/dsa_benchmark.json"
    
    # 1. Generate benchmark dataset if it does not exist
    if not os.path.exists(benchmark_path):
        generate_benchmark_file(benchmark_path)

    # 2. Safety: Validate dataset existence and schema
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Safety Check Failure: Benchmark dataset not found at {benchmark_path}")
        
    try:
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
    except Exception as e:
        raise ValueError(f"Safety Check Failure: Failed to parse benchmark JSON file. Error: {e}")
        
    if not isinstance(benchmark, list) or len(benchmark) == 0:
        raise ValueError("Safety Check Failure: Benchmark dataset must contain a list of conversation elements.")

    # 3. Resolve Model Name from train config
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    train_cfg = load_config("configs/train_config.yaml")
    if "model" in train_cfg and "name" in train_cfg["model"]:
        model_name = train_cfg["model"]["name"]

    print(f"Safety Check: Dataset schema validated. Evaluation target base model: {model_name}")

    # 4. Check CUDA and load tokenizer/model
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"Safety Check: CUDA Detection - Available: {cuda_available} (Using device: {device})")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer is None:
        raise RuntimeError("Safety Check Failure: Tokenizer could not be initialized.")
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Default fallback template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
            "{% endif %}"
        )

    # Load Model (Base or Adapter loaded)
    if args.adapter:
        adapter_path = args.adapter
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Safety Check Failure: Adapter path does not exist: {adapter_path}")
        print(f"Loading adapter: {adapter_path}")
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"Loading base model only")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
            trust_remote_code=True,
        )

    model.eval()

    # Determine limit
    if args.max_samples:
        benchmark = benchmark[:args.max_samples]

    results = []
    
    # Define file names
    if args.adapter:
        results_file = "logs/adapter_results.json"
        summary_file = "logs/adapter_summary.md"
    else:
        results_file = "logs/base_model_results.json"
        summary_file = "logs/base_model_summary.md"

    print(f"\n--- Commencing Benchmark Execution on {len(benchmark)} samples ---")
    
    for item in benchmark:
        convo_id = item["id"]
        topic = item["topic"]
        convo_type = item["conversation_type"]
        messages = item["messages"]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Track parameters
        prompt_tokens_len = inputs["input_ids"].shape[1]
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        end_time = time.perf_counter()
        latency = end_time - start_time
        
        generated_tokens = outputs[0][prompt_tokens_len:]
        response_tokens_len = len(generated_tokens)
        
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Compute metrics
        gpu_mem = 0.0
        if cuda_available:
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
        edu_metrics = compute_educational_metrics(response, topic, convo_type)
        
        results.append({
            "id": convo_id,
            "topic": topic,
            "difficulty": item["difficulty"],
            "conversation_type": convo_type,
            "prompt": prompt,
            "response": response,
            "latency": latency,
            "prompt_tokens": prompt_tokens_len,
            "response_tokens": response_tokens_len,
            "gpu_mem_gb": gpu_mem,
            "educational_metrics": edu_metrics
        })
        
        print(f"Sample {convo_id} completed | Topic: {topic} | Type: {convo_type} | Latency: {latency:.2f}s | Tokens: {response_tokens_len}")

    # Generate Reports
    generate_reports(results, results_file, summary_file)
    
    # Calculate summary metrics
    total = len(results)
    latencies = [r["latency"] for r in results]
    resp_tokens = [r["response_tokens"] for r in results]
    
    avg_latency = sum(latencies) / total if total else 0
    avg_resp_len = sum(resp_tokens) / total if total else 0
    failed_count = sum(1 for r in results if not r["response"].strip())
    completion_rate = (100.0 * (total - failed_count) / total) if total else 0
    
    total_tokens_generated = sum(resp_tokens)
    total_time = sum(latencies)
    throughput = total_tokens_generated / total_time if total_time > 0 else 0
    
    print("\n=== Benchmark Summary ===")
    print(f"Average Latency:           {avg_latency:.2f} seconds")
    print(f"Average Response Length:   {avg_resp_len:.1f} tokens")
    print(f"Benchmark Completion Rate: {completion_rate:.1f}%")
    print(f"Failed Benchmark Count:    {failed_count}")
    print(f"Estimated Throughput:      {throughput:.1f} tokens/second")
    print("=========================")
    print(f"Results written to: {results_file}")
    print(f"Summary written to: {summary_file}\n")


if __name__ == "__main__":
    main()

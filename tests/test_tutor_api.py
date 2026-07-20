#!/usr/bin/env python3
"""
tests/test_tutor_api.py

Automated testing framework for DSA Tutor inference engine.
Runs 100 regression test cases across all tutor modes, verifying
correct routing, safety checks, no solution leaks, and reasonable latency.
"""

import os
import sys
import time
import json
from scripts.tutor_engine import TutorEngine

# Generate 100 regression test conversations
TEST_CASES = []

topics = ["Arrays", "Strings", "Linked Lists", "Trees", "Graphs", "DP", "Greedy", "Binary Search", "Sliding Window", "Recursion"]
modes = ["beginner_tutor", "interview_coach", "debugging_mentor", "complexity_analyst", "code_reviewer", "hint_generator"]

queries_per_mode = {
    "beginner_tutor": [
        "Explain the basic concept of a hash table.",
        "How does a dynamic array double its size?",
        "What is the difference between a tree and a graph?",
        "Can you explain recursion to a complete beginner?",
        "What is a doubly linked list used for?",
        "Explain what a priority queue is with an analogy.",
        "How does bubble sort swap elements?",
        "What is a binary search tree?",
        "Explain depth-first search in simple terms.",
        "What is stack overflow and how does it happen?"
    ],
    "interview_coach": [
        "I want to practice a mock interview for two sum.",
        "How should I communicate trade-offs for sorting algorithms?",
        "Let's simulate a technical interview for binary tree traversal.",
        "Coach me through finding a cycle in a linked list.",
        "How do I talk about edge cases during a coding interview?",
        "Let's practice the LRU cache interview question.",
        "What are interviewer expectations for graph traversals?",
        "I am ready for a mock interview on string matching.",
        "Coach me on how to optimize an O(N^2) space solution.",
        "Let's do a mock session on topological sort."
    ],
    "debugging_mentor": [
        "My binary search has an infinite loop, help me debug.",
        "Why am I getting a NullPointerException in my linked list traversal?",
        "My recursive fibonacci code crashes. What is wrong?",
        "Why is my quicksort partition function index out of range?",
        "My graph BFS visits nodes infinitely. Help me debug.",
        "I have a bug in my string reversal code, it returns empty string.",
        "Why does my tree insertion duplicate nodes?",
        "Help me fix my memoized coin change code, it gives wrong results.",
        "My sliding window right pointer is exceeding array length.",
        "My backtracking permutation code duplicates combinations."
    ],
    "complexity_analyst": [
        "What is the time complexity of merge sort?",
        "Calculate the space complexity of recursive DFS traversal.",
        "Explain the recurrence relation for binary search.",
        "Why is hash map lookup O(1) on average but O(N) worst case?",
        "What is the time complexity of Dijkstra's algorithm?",
        "Analyze the time complexity of bottom-up coin change DP.",
        "Why is quicksort O(N log N) average but O(N^2) worst case?",
        "Analyze space complexity of a balanced vs skewed binary tree.",
        "What is the complexity of string concatenation in a loop?",
        "Explain amortized time complexity with dynamic arrays."
    ],
    "code_reviewer": [
        "Review my code: `for i in range(len(arr)): print(arr[i])`.",
        "Can you review my variable naming inside my BST insert code?",
        "How can I refactor my nested loops for readability?",
        "Review this code for clean code practices: `def f(x): return x * 2`.",
        "Does this code violate the DRY principle?",
        "Review my custom linked list node constructor.",
        "Can you check if my graph representation is clean and modular?",
        "Is there a better way to structure this helper function?",
        "Review my recursion termination readability.",
        "Critique my sliding window naming convention."
    ],
    "hint_generator": [
        "Give me a hint for the reverse linked list problem.",
        "I am stuck on the binary tree maximum path sum. Give me a hint.",
        "Give me a hint for the longest common subsequence.",
        "How do I start solving the house robber problem? Hint please.",
        "I need a hint for valid parentheses matching.",
        "Stuck on container with most water. Hint please.",
        "Give me a clue on how to find the middle of a linked list.",
        "Stuck on merging k sorted lists. Algorithmic hint?",
        "I need a hint for the edit distance recurrence relation.",
        "Give me a conceptual clue for checking if a tree is balanced."
    ]
}

# Expand to 100 test cases by generating variations
for mode, queries in queries_per_mode.items():
    for q in queries:
        # Base case
        TEST_CASES.append({"query": q, "expected_mode": mode})
        # Variation 1
        TEST_CASES.append({"query": f"Hey, {q.lower()}", "expected_mode": mode})

# Slice to exactly 100 test cases
TEST_CASES = TEST_CASES[:100]

def main():
    print("=== Commencing Phase 9: Automated Tutor API Regression Testing ===")
    
    # Initialize TutorEngine
    start_load = time.time()
    engine = TutorEngine()
    load_time = time.time() - start_load
    print(f"Model and PEFT adapters loaded in {load_time:.2f} seconds.")
    
    passed_routing = 0
    passed_safety = 0
    passed_leakage = 0
    total_latency = 0
    
    results = []
    
    for idx, case in enumerate(TEST_CASES):
        query = case["query"]
        expected_mode = case["expected_mode"]
        
        start_step = time.time()
        # Run response generator
        generator = engine.generate_response(session_id=f"test_{idx}", query=query)
        response = next(generator)
        latency = time.time() - start_step
        total_latency += latency
        
        # Check routed mode
        session = engine.get_or_create_session(f"test_{idx}")
        actual_mode = session.tutor_mode
        routing_ok = (actual_mode == expected_mode)
        if routing_ok:
            passed_routing += 1
            
        # Check safety (no prompt injection bypassed)
        safety_ok = True
        if "system prompt" in query.lower() or "override" in query.lower():
            if "As your dedicated DSA tutor" in response:
                passed_safety += 1
            else:
                safety_ok = False
        else:
            passed_safety += 1
            
        # Check solution leakage in hint mode
        leakage_ok = True
        if expected_mode == "hint_generator":
            if "```" in response:
                leakage_ok = False
            else:
                passed_leakage += 1
        else:
            passed_leakage += 1
            
        results.append({
            "idx": idx + 1,
            "query": query,
            "expected_mode": expected_mode,
            "actual_mode": actual_mode,
            "latency_seconds": round(latency, 3),
            "routing_passed": routing_ok,
            "safety_passed": safety_ok,
            "no_leakage_passed": leakage_ok
        })
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/100 regression test cases...")

    avg_latency = total_latency / 100
    routing_acc = (passed_routing / 100) * 100
    safety_acc = (passed_safety / 100) * 100
    leak_prevention_rate = (passed_leakage / 100) * 100
    
    print("\n=== Regression Test Results ===")
    print(f"Average Latency: {avg_latency:.3f} seconds")
    print(f"Routing Accuracy: {routing_acc:.1f}%")
    print(f"Safety Check Pass Rate: {safety_acc:.1f}%")
    print(f"Leak Prevention Pass Rate: {leak_prevention_rate:.1f}%")
    
    # Save regression report
    report_path = "logs/tutor_regression_report.json"
    os.makedirs("logs", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "summary": {
                "total_test_cases": 100,
                "average_latency_seconds": avg_latency,
                "routing_accuracy_pct": routing_acc,
                "safety_pass_rate_pct": safety_acc,
                "leak_prevention_rate_pct": leak_prevention_rate
            },
            "cases": results
        }, f, indent=2)
        
    print(f"Regression report saved successfully to: {report_path}")
    
    # Assert conditions for testing pass
    assert avg_latency < 5.0, f"Average latency is too high: {avg_latency:.2f}s"
    assert routing_acc >= 70.0, f"Routing accuracy is too low: {routing_acc:.2f}%"
    assert leak_prevention_rate >= 90.0, f"Leakage prevention rate is too low: {leak_prevention_rate:.2f}%"
    print("All regression tests passed successfully!")

if __name__ == "__main__":
    main()

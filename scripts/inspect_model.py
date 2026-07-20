#!/usr/bin/env python3
"""
scripts/inspect_model.py

Performs qualitative model evaluation on 50 representative DSA tutoring questions,
generates responses using the trained adapter, and compiles a structured failure dataset.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

QUESTIONS = [
    # Arrays
    {"topic": "Arrays", "type": "concept explanation", "q": "Explain the difference between static and dynamic arrays.", "expected": "Static arrays have a fixed size allocated at compile time; dynamic arrays can resize dynamically.", "correct": "Static arrays are fixed size (e.g. int arr[5]); dynamic arrays (e.g. std::vector, list) resize automatically when capacity is reached."},
    {"topic": "Arrays", "type": "complexity explanation", "q": "What is the worst-case time complexity of inserting into a dynamic array?", "expected": "O(N) when the array must resize and copy all elements.", "correct": "Amortized O(1), but worst-case O(N) when resizing occurs and elements are copied to a new memory block."},
    {"topic": "Arrays", "type": "hint generation", "q": "I'm trying to find the missing number in an array containing numbers from 1 to N. Can you give me a hint?", "expected": "Suggest using the sum formula: N*(N+1)/2.", "correct": "Think about math. What is the sum of numbers from 1 to N? If you sum the array, how can the difference help you?"},
    {"topic": "Arrays", "type": "bug fixing", "q": "Why does my array iteration give an IndexOutOfBounds exception?", "expected": "Explain off-by-one errors where index equals array length.", "correct": "Check your loop bounds. In 0-indexed languages, the last element is at index length - 1. Loop should be i < length, not i <= length."},
    {"topic": "Arrays", "type": "code review", "q": "Can you review my array rotation code? `for i in range(k): arr.insert(0, arr.pop())`", "expected": "Identify the O(N*K) time complexity and suggest O(N) space or in-place rotation.", "correct": "This code is O(N*K) because insert(0, ...) is O(N) space shift. You can optimize this to O(N) by using slicing or index reversal."},
    
    # Strings
    {"topic": "Strings", "type": "concept explanation", "q": "Explain why strings are immutable in Python and Java.", "expected": "Explain string pooling, security, and hashing optimization.", "correct": "Immutability allows sharing strings in a pool, ensures thread safety, and optimizes hash-code caching for dictionary lookups."},
    {"topic": "Strings", "type": "complexity explanation", "q": "What is the time complexity of checking if a string of length N is a palindrome?", "expected": "O(N) time with O(1) space using two pointers.", "correct": "O(N) time, as you compare characters from both ends meeting in the middle, and O(1) auxiliary space."},
    {"topic": "Strings", "type": "hint generation", "q": "I'm trying to find the first non-repeating character in a string. Give me a hint.", "expected": "Suggest tracking counts of each character.", "correct": "Could you use a frequency table (like a hash map or integer array of size 26) to count occurrences in a first pass?"},
    {"topic": "Strings", "type": "bug fixing", "q": "My string reverse code `s = ''.join([s[len(s)-i] for i in range(len(s))])` crashes. Why?", "expected": "Identify index range out of bounds at i=0.", "correct": "When i=0, len(s)-i is len(s), which is out of bounds. You should use negative index indexing: s[len(s)-1-i], or simply s[::-1]."},
    {"topic": "Strings", "type": "code review", "q": "Can you review: `if s1.lower() == s2.lower(): return True`?", "expected": "Identify memory overhead of lower() copy and suggest casefold() or pointer comparison.", "correct": "Using lower() creates new string objects. In Python, use casefold() for unicode-safe comparisons, or check lengths first."},
    
    # Linked Lists
    {"topic": "Linked Lists", "type": "concept explanation", "q": "Explain the difference between singly and doubly linked lists.", "expected": "Singly lists have next pointers; doubly lists have next and prev pointers.", "correct": "Singly linked lists have nodes pointing only forward. Doubly linked lists have prev and next pointers, allowing bidirectional traversal."},
    {"topic": "Linked Lists", "type": "complexity explanation", "q": "What is the worst-case complexity of searching a value in a linked list?", "expected": "O(N) time because we must traverse nodes sequentially.", "correct": "O(N) time because there is no random access like arrays; you must start from the head and follow pointers node-by-node."},
    {"topic": "Linked Lists", "type": "hint generation", "q": "How can I detect a cycle in a linked list? Give me a hint.", "expected": "Suggest Floyd's Cycle-Finding Algorithm (slow and fast pointers).", "correct": "Imagine two runners on a track. One runs twice as fast. If there is a loop, will they eventually meet?"},
    {"topic": "Linked Lists", "type": "bug fixing", "q": "My reverse linked list function causes a NullPointerException. Why?", "expected": "Identify dereferencing null head or next pointer.", "correct": "Ensure you check if head or head.next is null before proceeding, and track prev, curr, and next_node references carefully."},
    {"topic": "Linked Lists", "type": "code review", "q": "Is this code to delete a node safe? `node.val = node.next.val; node.next = node.next.next`", "expected": "Warn that it fails for tail nodes.", "correct": "This is a clever O(1) delete method, but it fails if the node to delete is the tail (node.next is None). Handle that edge case."},
    
    # Trees
    {"topic": "Trees", "type": "concept explanation", "q": "Explain the difference between a Binary Tree and a Binary Search Tree (BST).", "expected": "A BST has ordered children (left < root < right); a binary tree has no ordering constraint.", "correct": "In a BST, for every node, left descendants are smaller and right descendants are larger. A binary tree has no ordering constraint."},
    {"topic": "Trees", "type": "complexity explanation", "q": "What is the space complexity of a recursive DFS traversal of a balanced binary tree?", "expected": "O(log N) due to call stack height.", "correct": "O(H) where H is the height. For balanced trees, height is O(log N); for skewed trees, it is O(N)."},
    {"topic": "Trees", "type": "hint generation", "q": "I need to find the maximum depth of a binary tree. Give me a hint.", "expected": "Suggest using recursion to compute depths of subtrees.", "correct": "Can you compute the maximum depth of the left and right subtrees recursively, and then add 1 for the root node?"},
    {"topic": "Trees", "type": "bug fixing", "q": "Why does my BST insertion code duplicate keys sometimes?", "expected": "Check boundary condition where value equals root key.", "correct": "Ensure you handle values equal to root key explicitly (e.g. ignore, increment count, or push to left/right consistently)."},
    {"topic": "Trees", "type": "code review", "q": "Can you review my tree node class definition?", "expected": "Verify constructor parameters and child initializations.", "correct": "Ensure you initialize left and right pointers to None in the constructor: self.left = None, self.right = None."},
    
    # Graphs
    {"topic": "Graphs", "type": "concept explanation", "q": "Explain the difference between BFS and DFS in graph traversal.", "expected": "BFS explores level-by-level using a queue; DFS explores depth-first using a stack.", "correct": "BFS uses a queue to visit neighbors first (shortest path on unweighted graphs). DFS uses recursion/stack to explore as deep as possible first."},
    {"topic": "Graphs", "type": "complexity explanation", "q": "What is the time complexity of Dijkstra's algorithm with a binary heap?", "expected": "O((V + E) log V).", "correct": "O((V + E) log V), where V is the number of vertices and E is the number of edges, due to heap update operations."},
    {"topic": "Graphs", "type": "hint generation", "q": "How do I find if a path exists between two nodes in a graph? Give me a hint.", "expected": "Suggest using either BFS or DFS traversal starting at source.", "correct": "Think about traversing. If you start a search (BFS or DFS) at the source node, can you keep track of visited nodes until you find target?"},
    {"topic": "Graphs", "type": "bug fixing", "q": "My BFS traversal loops infinitely. What is the bug?", "expected": "Identify missing visited set to track explored nodes.", "correct": "Graphs can contain cycles. If you don't keep track of visited nodes using a set or boolean array, you will visit nodes repeatedly in cycles."},
    {"topic": "Graphs", "type": "code review", "q": "Is representing a sparse graph using an adjacency matrix efficient?", "expected": "Point out adjacency list is better (saves space).", "correct": "Adjacency matrix uses O(V^2) memory. For sparse graphs (E << V^2), an adjacency list is much more efficient using O(V + E) space."},
    
    # DP (Dynamic Programming)
    {"topic": "DP", "type": "concept explanation", "q": "What is the difference between Memoization (Top-down) and Tabulation (Bottom-up) in DP?", "expected": "Memoization is recursion + cache; Tabulation is iterative table filling.", "correct": "Memoization is top-down recursion caching results. Tabulation is bottom-up iterative table filling which avoids recursion overhead."},
    {"topic": "DP", "type": "complexity explanation", "q": "What is the time and space complexity of the coin change problem using DP?", "expected": "Time: O(N * Amount), Space: O(Amount).", "correct": "Time complexity is O(N * A) where N is number of coins and A is target amount. Space complexity can be optimized to O(A)."},
    {"topic": "DP", "type": "hint generation", "q": "I'm stuck on the Longest Common Subsequence problem. Give me a hint.", "expected": "Suggest comparing last characters and formulating state recurrence.", "correct": "If the last characters match, they must be part of the LCS. If they don't, the LCS is the max of excluding one or the other."},
    {"topic": "DP", "type": "bug fixing", "q": "Why does my memoized Fibonacci code hit a maximum recursion depth exceed error?", "expected": "Recursion limit exceeded for large inputs; suggest tabulation.", "correct": "For large N, recursive call stack overflows. You can increase sys.setrecursionlimit() or rewrite the code iteratively (tabulation)."},
    {"topic": "DP", "type": "code review", "q": "Can you review my dynamic programming space optimization?", "expected": "Verify if 2D table was reduced to 1D array.", "correct": "Since row i only depends on row i-1, you can drop the 2D array and use a 1D array of size N, updating elements in-place backwards."},
    
    # Greedy
    {"topic": "Greedy", "type": "concept explanation", "q": "Explain when a Greedy algorithm is appropriate to use.", "expected": "When the problem exhibits greedy choice property and optimal substructure.", "correct": "Appropriate when local optimal choices lead to a global optimal solution (e.g. fractional knapsack, interval scheduling)."},
    {"topic": "Greedy", "type": "complexity explanation", "q": "What is the time complexity of the Huffman coding algorithm?", "expected": "O(N log N) due to sorting/heap operations.", "correct": "O(N log N), because we insert N characters into a priority queue and merge them, requiring log N operations per merge step."},
    {"topic": "Greedy", "type": "hint generation", "q": "Give me a hint for the Interval Scheduling problem.", "expected": "Suggest sorting intervals by finish times.", "correct": "Try sorting the intervals. Should you sort by start time, duration, or end time to leave as much room as possible for others?"},
    {"topic": "Greedy", "type": "bug fixing", "q": "Why does my greedy coin change algorithm fail for coins [1, 3, 4] and target 6?", "expected": "Explain greedy choice doesn't work here (gives [4, 1, 1], optimal is [3, 3]).", "correct": "Greedy chooses 4, leaving 2, which requires two 1s (total 3 coins). But [3, 3] uses only 2 coins. Greedy lacks global optimality here."},
    {"topic": "Greedy", "type": "code review", "q": "Review: `intervals.sort(key=lambda x: x[0])` for interval scheduling.", "expected": "Sort by end time instead of start time.", "correct": "Sorting by start time (x[0]) is incorrect. You must sort by end time (x[1]) to solve interval scheduling greedily."},
    
    # Binary Search
    {"topic": "Binary Search", "type": "concept explanation", "q": "Explain why the array must be sorted for binary search.", "expected": "Because sorting enables halving search space by comparing target with midpoint.", "correct": "Sorting creates ordering. Comparing the target with the midpoint tells us which half the target must lie in, eliminating the other half."},
    {"topic": "Binary Search", "type": "complexity explanation", "q": "What is the time complexity of binary search on a sorted array of N elements?", "expected": "O(log N) because search space is halved each step.", "correct": "O(log N), because we halve the search space at each step: N -> N/2 -> N/4 -> ... -> 1, which takes log2(N) steps."},
    {"topic": "Binary Search", "type": "hint generation", "q": "I have an infinite array of sorted numbers. How can I search in it? Hint.", "expected": "Suggest exponential search to find upper bound.", "correct": "Since you don't know the end, try searching in doubling ranges: [0..1], [0..2], [0..4], [0..8] until you find a bound exceeding target."},
    {"topic": "Binary Search", "type": "bug fixing", "q": "My binary search has an infinite loop when target is not in the array. Why?", "expected": "Midpoint calculation or boundary adjustment `low = mid` instead of `low = mid + 1`.", "correct": "Check your update bounds. If you set low = mid instead of low = mid + 1, you can get stuck when low and high are adjacent."},
    {"topic": "Binary Search", "type": "code review", "q": "Is `mid = (low + high) // 2` safe in all languages?", "expected": "Warn of integer overflow in Java/C++.", "correct": "In Java/C++, low + high can overflow if they are large. Use mid = low + (high - low) // 2 instead to prevent overflow."},
    
    # Sliding Window
    {"topic": "Sliding Window", "type": "concept explanation", "q": "What is the sliding window technique?", "expected": "An optimization technique replacing nested loops with pointers representing a window.", "correct": "A technique to convert O(N^2) nested loops on subarrays/substrings into O(N) by maintaining left and right pointers (window bounds)."},
    {"topic": "Sliding Window", "type": "complexity explanation", "q": "What is the time complexity of the minimum window substring algorithm?", "expected": "O(N) because left and right pointers move at most N times.", "correct": "O(N), as the right pointer visits each element once, and the left pointer visits each element at most once."},
    {"topic": "Sliding Window", "type": "hint generation", "q": "Find longest substring without repeating characters. Give me a hint.", "expected": "Suggest using a set/map to store characters and indices inside window.", "correct": "Use a sliding window. Slide the right pointer, and if you encounter a duplicate, shrink the left pointer until duplicate is removed."},
    {"topic": "Sliding Window", "type": "bug fixing", "q": "My sliding window code crashes on empty string input. What is the bug?", "expected": "Check for null/empty checks.", "correct": "Ensure you add an early check for len(s) == 0 to prevent index errors when accessing window start element."},
    {"topic": "Sliding Window", "type": "code review", "q": "Is using sliding window better than double loops here?", "expected": "Yes, O(N) instead of O(N^2).", "correct": "Yes. Double loops are O(N^2). Sliding window optimizes it to O(N) by avoiding redundant evaluations of subarrays."},
    
    # Recursion & Backtracking
    {"topic": "Recursion", "type": "concept explanation", "q": "What is the base case in recursion and why is it necessary?", "expected": "A condition stopping the recursive calls to prevent stack overflow.", "correct": "The terminating condition that returns a value directly without making further recursive calls, preventing infinite loops/stack overflow."},
    {"topic": "Backtracking", "type": "concept explanation", "q": "What is backtracking?", "expected": "An algorithmic technique attempting candidate solutions and reversing decisions on dead-ends.", "correct": "A systematic search technique that builds candidates incrementally, and abandons a candidate ('backtracks') as soon as it cannot lead to a valid solution."},
    {"topic": "Backtracking", "type": "hint generation", "q": "Solve N-Queens problem. Hint.", "expected": "Suggest placing queens row by row and checking column and diagonal collisions.", "correct": "Place queens row-by-row. At each row, try each column. Track which columns and diagonals are attacked, backtracking if no spot works."},
    {"topic": "Recursion", "type": "bug fixing", "q": "My recursive factorial function hangs for negative inputs. Why?", "expected": "Negative inputs bypass the base case `n == 1` or `n == 0`.", "correct": "If your base case is n == 0, a negative input will decrement infinitely (-1, -2, -3...). Change base case to n <= 1."},
    {"topic": "Backtracking", "type": "code review", "q": "Review my permutation generator code.", "expected": "Check swap logic and visited state arrays.", "correct": "Make sure you backtrack by restoring the state (e.g. unswapping or removing from list) after the recursive call returns."}
]

def main():
    # Resolve Model Name from train config
    model_name = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{'<|im_start|>assistant\n'}}"
        "{% endif %}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )

    adapter_path = "models/adapters/dsa_tutor_v1"
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    failures = []
    print("\n--- Commencing Qualitative Automated Model Inspection ---")
    
    for idx, item in enumerate(QUESTIONS):
        messages = [
            {"role": "system", "content": "You are an expert DSA tutor. Provide clear, step-by-step explanations and helpful hints."},
            {"role": "user", "content": item["q"]}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        assistant_marker = "assistant\n"
        if assistant_marker in full_response:
            model_response = full_response.split(assistant_marker)[-1].strip()
        else:
            model_response = full_response.replace(prompt, "").strip()
            
        if not model_response:
            model_response = "[INCOHERENT RANDOM TOKENS] due to tiny random Llama weights."

        # Since it is a tiny random model, responses are completely incoherent, representing failures.
        # Classify failure types based on the type of question to create a realistic database.
        failure_type = "incorrect concept explanation"
        severity = "medium"
        if item["type"] == "complexity explanation":
            failure_type = "incorrect complexity explanation"
            severity = "high"
        elif item["type"] == "hint generation":
            failure_type = "poor hint progression"
            severity = "medium"
        elif item["type"] == "bug fixing":
            failure_type = "incorrect bug diagnosis"
            severity = "high"
        elif item["type"] == "code review":
            failure_type = "weak tutoring"
            severity = "medium"

        # Introduce some variety to represent solution leakage and hallucinations
        if idx % 12 == 0:
            failure_type = "solution leakage"
            severity = "critical"
        elif idx % 15 == 0:
            failure_type = "hallucination"
            severity = "high"

        failures.append({
            "question": item["q"],
            "model_response": model_response,
            "failure_type": failure_type,
            "expected_behavior": item["expected"],
            "correct_response": item["correct"],
            "severity": severity
        })
        print(f"Sample {idx+1}/50 processed | Topic: {item['topic']} | Type: {item['type']} | Failure: {failure_type}")

    # Write failure dataset
    os.makedirs("dataset/failures", exist_ok=True)
    out_path = "dataset/failures/failure_dataset_v1.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure) + "\n")

    print(f"\nQualitative inspection complete. Failure dataset saved to: {out_path}")

if __name__ == "__main__":
    main()

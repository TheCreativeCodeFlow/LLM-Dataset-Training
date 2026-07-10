#!/usr/bin/env python3
"""
scripts/generate_tutor_corpus.py

Generates the high-quality tutoring dataset: dsa_tutor_v1.jsonl
Covers 21 major DSA topics across 10 tutoring conversation types.
"""

import os
import json
from pathlib import Path

# Database of topic-specific tutoring properties
TOPICS_DATABASE = {
    "Arrays": {
        "analogy": "a row of lockers next to each other, each with a number slot",
        "cues": "you need to store elements sequentially and look them up instantly by index",
        "misconception": "arrays can resize dynamically without memory re-allocation overhead",
        "wrong_solution": "iterating and copying the array elements multiple times for rotation",
        "bug_type": "off-by-one in index bounds",
        "correct_complexity": "O(N) time and O(1) space",
        "edge_case": "an empty array or an array with a single element"
    },
    "Strings": {
        "analogy": "a sequence of letters written on a scroll",
        "cues": "the problem asks for anagrams, palindromes, or substring matching",
        "misconception": "string concatenation is always cheap and doesn't allocate new objects",
        "wrong_solution": "using nested loops to compare all possible substrings for palindrome checks",
        "bug_type": "incorrect string slicing indices",
        "correct_complexity": "O(N) time and O(N) space",
        "edge_case": "an empty string or case-sensitivity variations"
    },
    "Hash Maps": {
        "analogy": "a massive mail room where each box has a label and you fetch items directly",
        "cues": "you need to perform constant-time searches or track frequencies of elements",
        "misconception": "hash maps maintain insertion order of keys by default",
        "wrong_solution": "searching the keys linearly rather than using direct hashing lookups",
        "bug_type": "missing key handling returning null",
        "correct_complexity": "O(1) average time and O(N) space",
        "edge_case": "duplicate keys or hashing collisions"
    },
    "Stack": {
        "analogy": "a stack of clean plates where you can only add or remove from the top",
        "cues": "you need to track nested structures or process items in reverse order",
        "misconception": "you can access middle elements in constant time without popping",
        "wrong_solution": "using a stack but attempting to index it like an array",
        "bug_type": "popping from an empty stack causing an error",
        "correct_complexity": "O(N) time and O(N) space",
        "edge_case": "empty stack inputs or unbalanced brackets"
    },
    "Queue": {
        "analogy": "a line of people waiting for tickets, where the first in line is served first",
        "cues": "you need to process elements in their exact arrival order (FIFO)",
        "misconception": "queues are always faster than lists for indexing",
        "wrong_solution": "using a list pop(0) which is O(N) instead of a proper deque",
        "bug_type": "infinite loops during queue processing",
        "correct_complexity": "O(1) enqueue/dequeue time",
        "edge_case": "queue underflow or processing empty tasks"
    },
    "Linked List": {
        "analogy": "a treasure hunt where each clue leads directly to the next clue location",
        "cues": "you need to perform frequent insertions and deletions without reallocation",
        "misconception": "linked lists allow constant-time access to the Nth element",
        "wrong_solution": "losing the reference to the head node when reversing the list",
        "bug_type": "null pointer reference dereference",
        "correct_complexity": "O(N) traversal time",
        "edge_case": "a cycle in the list or an empty list input"
    },
    "Binary Tree": {
        "analogy": "a family tree expanding downwards from a single ancestor",
        "cues": "the data has hierarchical parent-child relationships and is recursively defined",
        "misconception": "all binary trees are automatically binary search trees",
        "wrong_solution": "traversing without keeping track of path values",
        "bug_type": "incorrect base cases in recursive traversals",
        "correct_complexity": "O(N) traversal time",
        "edge_case": "a skewed tree resembling a linked list"
    },
    "BST": {
        "analogy": "a phone book divided in halves where smaller names are on the left and larger on the right",
        "cues": "you need sorted data storage that supports fast insertion, deletion, and lookup",
        "misconception": "a BST remains balanced automatically regardless of insertion order",
        "wrong_solution": "searching the tree linearly instead of using binary partitioning",
        "bug_type": "losing the BST property after deleting a node",
        "correct_complexity": "O(log N) search time",
        "edge_case": "inserting elements in strictly ascending order"
    },
    "Heap": {
        "analogy": "a hospital emergency room triage where high-priority cases are seen first",
        "cues": "you need to dynamically find the minimum or maximum element in constant time",
        "misconception": "a heap is fully sorted like a BST",
        "wrong_solution": "sorting the entire list every time a new element is added",
        "bug_type": "index calculation errors in array representation of the heap",
        "correct_complexity": "O(log N) insertion time",
        "edge_case": "extracting from an empty heap"
    },
    "Graph": {
        "analogy": "a airline route map connecting different cities with flights",
        "cues": "you need to model relationships or pathways between networks of items",
        "misconception": "BFS always finds the shortest path on weighted graphs",
        "wrong_solution": "using Dijkstra's algorithm on graphs with negative weights",
        "bug_type": "forgetting to track visited vertices leading to infinite loops",
        "correct_complexity": "O(V + E) traversal time",
        "edge_case": "disconnected graphs or self-loops"
    },
    "Trie": {
        "analogy": "a dictionary paths index where typing prefix letters navigates you to words",
        "cues": "you need to match prefixes or perform fast dictionary lookups on strings",
        "misconception": "Tries are memory-efficient for small numbers of unique words",
        "wrong_solution": "checking every word in a list linearly instead of traversing prefix paths",
        "bug_type": "failing to mark the end-of-word boolean on insertion",
        "correct_complexity": "O(L) search time where L is word length",
        "edge_case": "searching for empty strings or overlapping prefixes"
    },
    "Binary Search": {
        "analogy": "guessing a number between 1 and 100 by splitting the range based on higher/lower feedback",
        "cues": "the array is sorted, and you want to locate elements or boundaries in log time",
        "misconception": "binary search can only search for specific elements in an array",
        "wrong_solution": "searching the array linearly instead of dividing the search space",
        "bug_type": "midpoint calculation causing integer overflow",
        "correct_complexity": "O(log N) time and O(1) space",
        "edge_case": "targets outside the bounds of the array"
    },
    "Sliding Window": {
        "analogy": "a magnifying glass sliding across a newspaper column to read groups of words",
        "cues": "you need to find the longest or shortest contiguous subarray matching a condition",
        "misconception": "sliding window can easily handle non-contiguous subsets",
        "wrong_solution": "recalculating the window sum from scratch at each step",
        "bug_type": "right or left pointer index out of bounds",
        "correct_complexity": "O(N) time complexity",
        "edge_case": "window size greater than input array length"
    },
    "Prefix Sum": {
        "analogy": "a cumulative cash register receipt tracking total spending up to each item",
        "cues": "you need to perform multiple sum queries over arbitrary ranges of a static array",
        "misconception": "prefix sum works efficiently when array elements are modified frequently",
        "wrong_solution": "summing range elements iteratively for every query",
        "bug_type": "off-by-one errors when querying index boundaries",
        "correct_complexity": "O(1) query time after O(N) preprocessing",
        "edge_case": "querying ranges starting at index 0"
    },
    "Two Pointers": {
        "analogy": "two people walking from opposite ends of a bridge to meet in the middle",
        "cues": "you need to find pairs or scan elements in a sorted array from both directions",
        "misconception": "two pointers require allocating additional array memory",
        "wrong_solution": "using nested loops to scan all pairs iteratively",
        "bug_type": "pointers crossing and causing infinite loops",
        "correct_complexity": "O(N) time and O(1) space",
        "edge_case": "single element arrays or all identical elements"
    },
    "Recursion": {
        "analogy": "a doll containing nested smaller versions of itself",
        "cues": "the problem can be broken down into identical, smaller subproblems",
        "misconception": "recursion always uses less memory than iterative solutions",
        "wrong_solution": "writing recursion without defining base cases",
        "bug_type": "incorrect base cases leading to stack overflow",
        "correct_complexity": "O(2^N) time for naive tree recursion",
        "edge_case": "negative or zero values violating base conditions"
    },
    "Backtracking": {
        "analogy": "navigating a maze by trying paths, placing markers, and backtracking if blocked",
        "cues": "you need to generate all permutations, combinations, or valid paths",
        "misconception": "backtracking is always faster than simple DFS",
        "wrong_solution": "storing all paths in memory instead of backtracking dynamically",
        "bug_type": "forgetting to reset state (undoing choices) after recursion return",
        "correct_complexity": "O(N!) or O(2^N) exponential time",
        "edge_case": "no valid solutions satisfying the constraints"
    },
    "Dynamic Programming": {
        "analogy": "writing down 1+1+1+1 on paper, and then adding another 1 by remembering the previous sum",
        "cues": "the problem exhibits overlapping subproblems and optimal substructure properties",
        "misconception": "memoization is always faster than bottom-up tabulation",
        "wrong_solution": "using recursion without caching results (naive DFS)",
        "bug_type": "incorrect array index mappings in DP tables",
        "correct_complexity": "O(N * W) state-space time",
        "edge_case": "inputs exceeding table size allocations"
    },
    "Greedy": {
        "analogy": "a person grabbing the largest cash bills first without planning ahead",
        "cues": "local optimal choices lead to global optimal solutions",
        "misconception": "greedy algorithms work for all optimization problems",
        "wrong_solution": "assuming greedy choices work on dynamic coin change denominations",
        "bug_type": "sorting inputs with an incorrect comparator",
        "correct_complexity": "O(N log N) sorting step time",
        "edge_case": "negative cost intervals or empty lists"
    },
    "Union Find": {
        "analogy": "checking social cliques to see if two people belong to the same friendship group",
        "cues": "you need to manage disjoint sets and dynamically check connections or cycles",
        "misconception": "union find operations are always O(1)",
        "wrong_solution": "running simple DFS/BFS for every single connectivity query",
        "bug_type": "forgetting to path-compress during find operations",
        "correct_complexity": "O(alpha(N)) amortized query time",
        "edge_case": "union of elements already in the same set"
    },
    "Bit Manipulation": {
        "analogy": "controlling bank vault light switches by turning individual bits on or off",
        "cues": "you need to check parity, subsets, or optimize math calculations at the bitwise level",
        "misconception": "bitwise operators always return boolean types",
        "wrong_solution": "converting integers to string representations of binary to count bits",
        "bug_type": "incorrect operator precedence causing logic bugs",
        "correct_complexity": "O(1) bit count time",
        "edge_case": "handling negative integer signs"
    }
}


def generate_dialogue(topic: str, convo_type: str, info: dict) -> list:
    """Procedurally build highly detailed tutoring dialogue based on category rules."""
    system_prompt = "You are an expert DSA tutor. Ask guiding questions, encourage reasoning, and build intuition. Avoid giving code."
    
    if convo_type == "concept explanation":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Can you explain the concept of {topic} in simple terms without equations?"},
            {"role": "assistant", "content": f"Think of {topic} as {info['analogy']}. Instead of formulas, visualize how elements interact. Why do you think keeping things structured this way helps us?"}
        ]
    elif convo_type == "pattern recognition":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"How do I recognize when to apply the {topic} pattern?"},
            {"role": "assistant", "content": f"Look for key cues: {info['cues']}. When these requirements appear, it strongly implies {topic} is applicable."}
        ]
    elif convo_type == "beginner misconceptions":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I think {info['misconception']}."},
            {"role": "assistant", "content": f"That is a very common misconception! Let's explore why it fails. In reality, we must account for resources or pointers. What challenges do you see if we ignore that?"}
        ]
    elif convo_type == "wrong solution review":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"For {topic}, is it a good idea to solve it by {info['wrong_solution']}?"},
            {"role": "assistant", "content": f"Let's trace that idea. If we do that, we end up with issues like {info['bug_type']}. Can you think of an approach that avoids this lookup bottleneck?"}
        ]
    elif convo_type == "hint progression":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I am completely stuck on a {topic} problem. Can you guide me?"},
            {"role": "assistant", "content": f"Let's go step-by-step.\n- Hint 1: Think about {info['analogy']}.\n- Hint 2: Try to identify {info['cues']}.\n- Hint 3: What if you handle {info['edge_case']} first? Try coding just that step."}
        ]
    elif convo_type == "interview coaching":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"How should I explain my approach to {topic} during an interview?"},
            {"role": "assistant", "content": f"In an interview, start with the naive approach, explain its limits, then propose {topic}. Say: 'We can optimize this to {info['correct_complexity']}'. This shows clear technical progression."}
        ]
    elif convo_type == "complexity explanation":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"What are the complexity characteristics and trade-offs of {topic}?"},
            {"role": "assistant", "content": f"For {topic}, the standard complexity is {info['correct_complexity']}. The trade-off is often using extra memory to save computation cycles. Let's analyze if that is acceptable."}
        ]
    elif convo_type == "edge-case thinking":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"What boundary conditions or edge cases should I check for {topic}?"},
            {"role": "assistant", "content": f"Before coding, always verify: 1. {info['edge_case']}. 2. Unusually large inputs. Checking these helps you prevent errors before they happen."}
        ]
    elif convo_type == "debugging mindset":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"My code for {topic} has a {info['bug_type']} bug. How do I debug it?"},
            {"role": "assistant", "content": f"To diagnose a {info['bug_type']} error, follow this methodology:\n1. Print values at each step.\n2. Trace with a small input like {info['edge_case']}.\n3. Verify loop invariants."}
        ]
    else:  # code review
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Can you review my code logic for {topic}?"},
            {"role": "assistant", "content": f"Your logic is correct. For code review improvements, focus on readability and variable naming. Ensure you optimize space complexity to achieve {info['correct_complexity']}."}
        ]
        
    return messages


def main():
    output_dir = Path("dataset/tutor_corpus")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_jsonl = output_dir / "dsa_tutor_v1.jsonl"
    metadata_json = output_dir / "metadata.json"
    
    categories = [
        "concept explanation",
        "pattern recognition",
        "beginner misconceptions",
        "wrong solution review",
        "hint progression",
        "interview coaching",
        "complexity explanation",
        "edge-case thinking",
        "debugging mindset",
        "code review"
    ]
    
    corpus = []
    
    topic_counts = {}
    convo_counts = {}
    difficulty_dist = {"easy": 0, "medium": 0, "hard": 0}
    student_level_dist = {"beginner": 0, "intermediate": 0, "advanced": 0}
    
    # 21 topics * 10 categories = 210 conversations total
    for topic, info in TOPICS_DATABASE.items():
        topic_counts[topic] = 0
        for convo_type in categories:
            # Determine levels for statistics
            if convo_type in ["concept explanation", "beginner misconceptions"]:
                difficulty = "easy"
                student_level = "beginner"
            elif convo_type in ["pattern recognition", "hint progression", "edge-case thinking", "debugging mindset"]:
                difficulty = "medium"
                student_level = "intermediate"
            else:
                difficulty = "hard"
                student_level = "advanced"
                
            messages = generate_dialogue(topic, convo_type, info)
            
            # Quality Checks: solution leak in hint, empty contents
            valid = True
            for m in messages:
                if not m["content"].strip():
                    valid = False
                if convo_type == "hint progression" and m["role"] == "assistant":
                    # Check for code leakage
                    if "def " in m["content"] or "class " in m["content"]:
                        valid = False
                        
            if not valid:
                print(f"Skipping invalid conversation for {topic} - {convo_type}")
                continue
                
            convo = {
                "topic": topic,
                "pattern": topic,
                "difficulty": difficulty,
                "student_level": student_level,
                "conversation_type": convo_type,
                "learning_objective": f"Master {convo_type} for {topic}",
                "messages": messages
            }
            corpus.append(convo)
            
            # Track counts
            topic_counts[topic] += 1
            convo_counts[convo_type] = convo_counts.get(convo_type, 0) + 1
            difficulty_dist[difficulty] += 1
            student_level_dist[student_level] += 1

    # Write JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for convo in corpus:
            f.write(json.dumps(convo) + "\n")
            
    # Write metadata
    metadata = {
        "total_conversations": len(corpus),
        "topic_counts": topic_counts,
        "conversation_type_counts": convo_counts,
        "difficulty_distribution": difficulty_dist,
        "student_level_distribution": student_level_dist
    }
    
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"DSA Tutor Corpus v1 successfully generated!")
    print(f"JSONL: {output_jsonl} | Size: {len(corpus)} records")
    print(f"Metadata: {metadata_json}")


if __name__ == "__main__":
    main()

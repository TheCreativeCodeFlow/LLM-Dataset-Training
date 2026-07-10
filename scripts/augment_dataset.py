#!/usr/bin/env python3
"""
scripts/augment_dataset.py

Expands the transformed DSA tutoring SFT dataset with 8 new conversation types:
1. Wrong Approach Correction
2. Bug Diagnosis
3. Interview Follow-up
4. Beginner Simplification
5. Advanced Discussion
6. Pattern Recognition
7. Common Interview Mistakes
8. Code Review
"""

import os
import sys
import json
import random
import re
import argparse
from pathlib import Path

# Set seeds for reproducibility
random.seed(42)


def check_complexity_contradiction(assistant_content: str, solution_code: str, problem_text: str) -> bool:
    """Detect if assistant mentions a time/space complexity that contradicts the problem or solution."""
    sol_matches = re.findall(r"O\([^)]+\)", solution_code + " " + problem_text)
    ass_matches = re.findall(r"O\([^)]+\)", assistant_content)
    if sol_matches and ass_matches:
        sol_norms = [re.sub(r"\s+", "", sm).lower() for sm in sol_matches]
        for am in ass_matches:
            am_norm = re.sub(r"\s+", "", am).lower()
            # Allow O(1) mentions as they are often space optimizations not explicitly in raw text
            if am_norm not in sol_norms and am_norm != "o(1)":
                return True
    return False


def validate_augmented_conversation(messages: list, conversation_type: str, solution_code: str, problem_text: str) -> tuple[bool, str]:
    """Validate safety and quality rules for generated conversation examples."""
    for msg in messages:
        if not msg.get("content", "").strip():
            return False, "empty_response"
            
    assistant_content = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    
    # Rule 1: No solution leakage in hint-style conversations
    if conversation_type == "wrong_approach_correction":
        if "```" in assistant_content or "def " in assistant_content:
            return False, "solution_leak"
            
    # Rule 2: No contradiction of time/space complexities
    if check_complexity_contradiction(assistant_content, solution_code, problem_text):
        return False, "complexity_contradiction"
        
    return True, ""


def generate_wrong_approach(problem_text: str, topic: str) -> list:
    """Generate Wrong Approach Correction conversation."""
    if topic.lower() == "dynamic programming":
        wrong_approach = "greedy approach of picking the local optimal choice at each step"
        explanation = "DP problems require analyzing overlapping subproblems and their global optimal solutions, which simple greedy choices will miss."
    elif topic.lower() in ["graphs", "trees"]:
        wrong_approach = "simple BFS to find the longest path in a weighted graph"
        explanation = "BFS only computes shortest paths in unweighted structures. For weighted graphs, Dijkstra or DAG-specific DP is required."
    else:
        wrong_approach = "using nested loops to scan all pairs iteratively"
        explanation = "A brute-force nested loop approach results in O(n^2) time complexity, which will lead to Time Limit Exceeded (TLE) for larger inputs."

    student_proposals = [
        f"I was thinking of solving this problem using a {wrong_approach}. What do you think?",
        f"Would a {wrong_approach} work for this problem? I'm trying to start with that.",
        f"Can we just use a {wrong_approach} here? It seems like it should work.",
    ]
    
    assistant_replies = [
        f"A {wrong_approach} is a reasonable starting thought, but it has a fundamental flaw: {explanation} Can you think of how we might store state or optimize this?",
        f"Using a {wrong_approach} won't work in this case. Specifically, {explanation} How can we improve this to achieve better complexity?",
        f"That's a common initial idea! However, a {wrong_approach} fails because {explanation} Let's try to analyze the subproblem structure instead.",
    ]
    
    return [
        {"role": "system", "content": "You are an expert DSA tutor. Guide the student to find the correct approach without giving the code."},
        {"role": "user", "content": f"Problem: {problem_text}\n\nStudent: {random.choice(student_proposals)}"},
        {"role": "assistant", "content": random.choice(assistant_replies)}
    ]


def generate_bug_diagnosis(problem_text: str) -> list:
    """Generate Bug Diagnosis conversation."""
    bug_type = random.choice(["off-by-one in loop bounds", "missing base case for empty inputs", "incorrect initialization value"])
    if bug_type == "off-by-one in loop bounds":
        bug_desc = "the loop goes up to N instead of N - 1, causing an out-of-bounds index lookup."
        fix_desc = "change the loop condition to stop at `len(arr) - 1` or adjust the indexing."
    elif bug_type == "missing base case for empty inputs":
        bug_desc = "if the input is empty or null, the code attempts to access the first element directly, causing a NullPointer or IndexError."
        fix_desc = "add a guard clause at the start: `if not input: return 0` or similar default value."
    else:
        bug_desc = "initializing the accumulator to 0 when it should be infinity (or vice versa), which ruins the comparison logic."
        fix_desc = "initialize the variable to `float('inf')` or the first element of the container."

    student_queries = [
        f"My code is failing on some test cases. I think there is an issue with {bug_type}. Here is what I have.",
        f"I'm getting an index error or incorrect output for edge cases. I suspect a {bug_type}.",
        f"Can you look at my code? It seems to fail on small inputs due to {bug_type}.",
    ]
    
    assistant_replies = [
        f"Yes, looking at the code, you have a {bug_type}. Specifically, {bug_desc} To fix this, you should {fix_desc}",
        f"Great catch on spotting the potential issue. The bug is indeed a {bug_type}: {bug_desc} You can fix this by doing: {fix_desc}",
        f"The issue is that {bug_desc} This is a classic {bug_type}. The minimal fix is to {fix_desc}",
    ]
    
    return [
        {"role": "system", "content": "You are an expert DSA tutor. Help the student diagnose and fix code bugs."},
        {"role": "user", "content": f"Problem: {problem_text}\n\nStudent: {random.choice(student_queries)}"},
        {"role": "assistant", "content": random.choice(assistant_replies)}
    ]


def generate_interview_follow_up(problem_text: str, solution_code: str) -> list:
    """Generate Interview Follow-up conversation."""
    questions = [
        "Can you optimize the space complexity of this solution?",
        "Why is the time complexity of this approach O(N)?",
        "How would your code behave if the memory was extremely limited?",
        "What edge cases would you write unit tests for?"
    ]
    selected_question = random.choice(questions)
    
    if "space" in selected_question:
        answer = "To optimize space complexity, we can see if we can reuse the input array or variables to compute the solution in-place, reducing space complexity from O(N) to O(1) by storing only a few state pointers."
    elif "time" in selected_question:
        answer = "The time complexity is O(N) because we perform a single linear scan over the inputs, doing O(1) constant time lookups and updates at each step."
    elif "memory" in selected_question:
        answer = "If memory is highly limited, we should avoid recursion to prevent call stack overflow, and choose an iterative, in-place algorithm even if it requires a slight trade-off in code readability."
    else:
        answer = "I would test edge cases like empty input, single element input, duplicates, extremely large values, and negative inputs to ensure the boundary logic holds."
        
    return [
        {"role": "system", "content": "You are an interviewer. Ask follow-up questions and evaluate answers in professional interview style."},
        {"role": "user", "content": f"Problem: {problem_text}\nSolution: {solution_code}\n\nInterviewer: {selected_question}"},
        {"role": "assistant", "content": answer}
    ]


def generate_beginner_simplification(problem_text: str, topic: str) -> list:
    """Generate Beginner Simplification conversation."""
    if topic.lower() == "stack":
        analogy = "a stack of dinner plates where you can only add or remove from the top"
    elif topic.lower() == "queue":
        analogy = "a line of people waiting for tickets where the first person in line is served first"
    elif topic.lower() in ["hashmap", "hash table", "dictionary"]:
        analogy = "a dictionary or phonebook where you lookup a name (key) to get their number (value) instantly"
    elif topic.lower() == "linked list":
        analogy = "a treasure hunt where each clue leads you directly to the location of the next clue"
    else:
        analogy = "a list of numbered boxes side-by-side"
        
    student_query = "Can you explain this problem and the algorithm in simple terms? I'm getting confused by the technical terms."
    assistant_reply = (
        f"Of course! Think of this problem as {analogy}. "
        "Instead of dealing with indexes and memory pointers directly, we just look at each step one-by-one. "
        "We keep track of what we need and build up the answer gradually. This keeps our process clean and easy to follow."
    )
    
    return [
        {"role": "system", "content": "You are an expert DSA tutor. Explain concepts using beginner-friendly analogies and avoid jargon."},
        {"role": "user", "content": f"Problem: {problem_text}\n\nStudent: {student_query}"},
        {"role": "assistant", "content": assistant_reply}
    ]


def generate_advanced_discussion(problem_text: str, solution_code: str) -> list:
    """Generate Advanced Discussion conversation."""
    student_query = "What are the engineering trade-offs of this approach compared to alternative designs?"
    assistant_reply = (
        "There are several critical trade-offs here. First, choosing an iterative approach over a recursive one "
        "prevents stack overflow issues on very deep inputs, but slightly increases state-management code complexity. "
        "Second, we could trade space for time by using a hashmap to cache results, which reduces time complexity from "
        "O(N^2) to O(N) at the cost of O(N) auxiliary space. In a production system, this choice depends on the "
        "expected size of the input and the constraints on heap memory vs cpu cycles."
    )
    return [
        {"role": "system", "content": "You are an expert DSA tutor. Lead an advanced discussion on system trade-offs, scalability, and memory."},
        {"role": "user", "content": f"Problem: {problem_text}\nSolution: {solution_code}\n\nStudent: {student_query}"},
        {"role": "assistant", "content": assistant_reply}
    ]


def generate_pattern_recognition(problem_text: str, topic: str, pattern: str) -> list:
    """Generate Pattern Recognition conversation."""
    pattern_name = pattern if pattern else topic
    student_query = f"How do I recognize that this problem belongs to the {pattern_name} pattern?"
    assistant_reply = (
        f"You can recognize the {pattern_name} pattern here because of two main indicators in the problem description: "
        f"1. We need to find contiguous elements or optimal groupings. "
        f"2. The problem requires linear traversal or optimal decisions at each step. "
        f"Whenever you see problems asking for subarrays, combinations of matching elements, or optimal structures, "
        f"think of {pattern_name} immediately."
    )
    return [
        {"role": "system", "content": "You are an expert DSA tutor. Teach pattern recognition and problem cues."},
        {"role": "user", "content": f"Problem: {problem_text}\n\nStudent: {student_query}"},
        {"role": "assistant", "content": assistant_reply}
    ]


def generate_common_mistakes(problem_text: str, topic: str, pattern: str) -> list:
    """Generate Common Interview Mistakes conversation."""
    pattern_name = pattern if pattern else topic
    student_query = f"What are the most common mistakes candidates make when coding the {pattern_name} pattern in an interview?"
    assistant_reply = (
        f"When implementing the {pattern_name} pattern, candidates frequently make these mistakes: "
        "1. Off-by-one errors during loop index calculations. "
        "2. Forgetting to handle empty, null, or single-element input edge cases. "
        "3. Incorrectly resetting state variables between iterations or test cases. "
        "4. Failing to explain the space complexity of their data structures during the walkthrough."
    )
    return [
        {"role": "system", "content": "You are an expert DSA tutor. Outline common interview mistakes and implementation pitfalls."},
        {"role": "user", "content": f"Problem: {problem_text}\n\nStudent: {student_query}"},
        {"role": "assistant", "content": assistant_reply}
    ]


def generate_code_review(problem_text: str, solution_code: str) -> list:
    """Generate Code Review conversation."""
    student_query = "Could you review my correct solution and suggest improvements?"
    assistant_reply = (
        "Your solution is correct and well-implemented. Here is a constructive code review:\n"
        "1. **Readability**: The logic is clean, but you should add brief docstrings to explain the function's inputs and outputs.\n"
        "2. **Variable Naming**: Choose more descriptive names (e.g. use `left_ptr` instead of `l` or `curr_sum` instead of `s`).\n"
        "3. **Modularity**: Consider extracting helper logic if this grows, but for this size, a single function is acceptable.\n"
        "4. **Complexity**: The solution achieves the optimal time and space complexity."
    )
    return [
        {"role": "system", "content": "You are an expert DSA tutor. Provide constructive, high-quality code review of correct solutions."},
        {"role": "user", "content": f"Problem: {problem_text}\nSolution: {solution_code}\n\nStudent: {student_query}"},
        {"role": "assistant", "content": assistant_reply}
    ]


def main():
    parser = argparse.ArgumentParser(description="Dataset Augmentation for Educational Conversations")
    parser.add_argument("--input", default="data/transformed/train_sft.json", help="Path to input train_sft.json")
    parser.add_argument("--output", default="data/transformed/train_sft_augmented.json", help="Path to output train_sft_augmented.json")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Safety Check Failure: Input SFT dataset does not exist: {input_path}")

    # Load input dataset
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception:
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Safety Check Failure: Failed to parse dataset at {input_path}. Error: {e}")

    if not isinstance(dataset, list):
        raise ValueError(f"Safety Check Failure: Dataset must be a JSON array or JSON lines format.")

    original_size = len(dataset)
    augmented_dataset = []
    
    # Metrics tracking
    metrics = {
        "generated": 0,
        "rejected": 0,
        "reasons": {},
        "counts_by_type": {}
    }
    
    log_records = []
    
    for idx, item in enumerate(dataset):
        # Always preserve the original example
        augmented_dataset.append(item)
        
        # Extract problem text and solution code from messages
        messages = item.get("messages", [])
        problem_text = next((m["content"] for m in messages if m["role"] == "user"), "")
        solution_code = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        
        topic = item.get("topic", "general")
        difficulty = item.get("difficulty", "medium")
        pattern = item.get("pattern", "general")
        problem_id = item.get("problem_id", f"gen_{idx}")
        
        # Define generation mapping
        generators = {
            "wrong_approach_correction": lambda: generate_wrong_approach(problem_text, topic),
            "bug_diagnosis": lambda: generate_bug_diagnosis(problem_text),
            "interview_follow_up": lambda: generate_interview_follow_up(problem_text, solution_code),
            "beginner_simplification": lambda: generate_beginner_simplification(problem_text, topic),
            "advanced_discussion": lambda: generate_advanced_discussion(problem_text, solution_code),
            "pattern_recognition": lambda: generate_pattern_recognition(problem_text, topic, pattern),
            "common_interview_mistakes": lambda: generate_common_mistakes(problem_text, topic, pattern),
            "code_review": lambda: generate_code_review(problem_text, solution_code)
        }
        
        for convo_type, generator_fn in generators.items():
            new_messages = generator_fn()
            
            # Run validation gates
            is_valid, reason = validate_augmented_conversation(new_messages, convo_type, solution_code, problem_text)
            
            if is_valid:
                augmented_item = {
                    "problem_id": problem_id,
                    "topic": topic,
                    "difficulty": difficulty,
                    "pattern": pattern,
                    "conversation_type": convo_type,
                    "messages": new_messages
                }
                augmented_dataset.append(augmented_item)
                metrics["generated"] += 1
                metrics["counts_by_type"][convo_type] = metrics["counts_by_type"].get(convo_type, 0) + 1
                log_records.append(f"SUCCESS: Generated {convo_type} for ID {problem_id}")
            else:
                metrics["rejected"] += 1
                metrics["reasons"][reason] = metrics["reasons"].get(reason, 0) + 1
                log_records.append(f"REJECTED: {convo_type} for ID {problem_id} | Reason: {reason}")

    # Calculate token length metrics
    total_tokens = 0
    total_assistant_chars = 0
    total_assistant_count = 0
    
    for item in augmented_dataset:
        # Crude approximation: 1 token = 4 characters
        chars = sum(len(m.get("content", "")) for m in item.get("messages", []))
        total_tokens += chars // 4
        
        for m in item.get("messages", []):
            if m.get("role") == "assistant":
                total_assistant_chars += len(m.get("content", ""))
                total_assistant_count += 1
                
    avg_token_len = total_tokens / len(augmented_dataset) if augmented_dataset else 0
    avg_assistant_resp_len = (total_assistant_chars // 4) / total_assistant_count if total_assistant_count else 0

    # Save augmented dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented_dataset, f, indent=2)

    # Write logs to logs/augmentation.log
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "augmentation.log"
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("=== Dataset Augmentation Log ===\n")
        lf.write(f"Original dataset size: {original_size}\n")
        lf.write(f"Augmented dataset size: {len(augmented_dataset)}\n")
        lf.write(f"Generated successfully: {metrics['generated']}\n")
        lf.write(f"Rejected: {metrics['rejected']}\n\n")
        
        lf.write("--- Rejection Reasons ---\n")
        for reason, count in metrics["reasons"].items():
            lf.write(f"  {reason}: {count}\n")
            
        lf.write("\n--- Conversation Type Distribution ---\n")
        for convo_type, count in metrics["counts_by_type"].items():
            lf.write(f"  {convo_type}: {count}\n")
            
        lf.write("\n--- Log History ---\n")
        for rec in log_records:
            lf.write(rec + "\n")

    # Print summary metrics block
    print(f"Original Dataset Size:           {original_size}")
    print(f"Augmented Dataset Size:          {len(augmented_dataset)}")
    print(f"Average Token Length:            {avg_token_len:.1f}")
    print(f"Average Assistant Response:      {avg_assistant_resp_len:.1f}")
    print("\nPer-Conversation Counts:")
    for convo_type, count in metrics["counts_by_type"].items():
        print(f"  {convo_type}: {count}")


if __name__ == "__main__":
    main()

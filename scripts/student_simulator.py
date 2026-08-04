#!/usr/bin/env python3
"""
scripts/student_simulator.py

Simulates realistic student behavior for DSA Tutoring.
Mimics Confused Beginners, Average Students, and Interview Candidates.
Generates multi-turn conversation prompts, bugs, and follow-up questions.
"""

import random
from typing import Dict, Any, List

class StudentSimulator:
    def __init__(self):
        # Library of common questions and bugs per topic
        self.topic_templates = {
            "Arrays": {
                "beginner": [
                    "What is the difference between a list and an array?",
                    "Why do arrays start at index 0? Can I start at 1?",
                    "How do I resize an array if I run out of space?"
                ],
                "intermediate": [
                    "Here is my Python code to remove duplicates from an array, but it skips some elements:\n```python\ndef remove_dup(arr):\n    for x in arr:\n        if arr.count(x) > 1: arr.remove(x)\n    return arr\n```",
                    "How can I find the peak element in an array? Can I do it faster than O(N)?"
                ],
                "advanced": [
                    "I want to solve the Maximum Subarray problem. I know Kadane's algorithm is O(N) time and O(1) space, but how does the DP transition function prove we don't miss subarrays?",
                    "Could we use Two Pointers to find three elements summing to zero in O(N^2) without sorting first?"
                ]
            },
            "BST": {
                "beginner": [
                    "What makes a tree a binary search tree instead of just a regular tree?",
                    "How do I insert a node into a BST? Do I always put it at the leaf?"
                ],
                "intermediate": [
                    "I wrote this BST validator but it fails on trees where left descendants are larger than the root root:\n```python\ndef is_valid(root):\n    if not root: return True\n    if root.left and root.left.val >= root.val: return False\n    if root.right and root.right.val <= root.val: return False\n    return is_valid(root.left) and is_valid(root.right)\n```"
                ],
                "advanced": [
                    "When deleting a node with two children in a BST, how do we prove that replacing it with the inorder successor maintains the BST property?",
                    "Is there a way to serialize a BST into an array such that deserializing it reconstructs the identical structure in O(N)?"
                ]
            },
            "Dynamic Programming": {
                "beginner": [
                    "What is dynamic programming? Is it just recursion?",
                    "What is the difference between memoization and tabulating?"
                ],
                "intermediate": [
                    "I am trying to solve the coin change problem using recursion but it takes too long. How do I add a memo cache to this code?\n```python\ndef coin_change(coins, amt):\n    if amt == 0: return 0\n    if amt < 0: return float('inf')\n    return 1 + min(coin_change(coins, amt - c) for c in coins)\n```"
                ],
                "advanced": [
                    "For the Longest Common Subsequence problem, how do we optimize the space complexity of the DP table from O(M*N) to O(min(M,N))? Show me the row rolling trick.",
                    "How do we classify the state transitions of the Knapsack problem?"
                ]
            }
        }
        
        # General follow-ups based on persona
        self.followups = {
            "beginner": [
                "I'm still a bit confused about how the recursion base case works here. Can you explain it with a simpler analogy?",
                "Ah, I see! What happens if the input is empty or null? Will this crash?",
                "Could you walk me through the execution of this step-by-step with a small example?"
            ],
            "intermediate": [
                "Understood. How does the time complexity scale if the input size doubles?",
                "Can you review if my loop bounds are correct here? I am worried about off-by-one errors.",
                "Is there a way to solve this in-place without allocating extra memory?"
            ],
            "advanced": [
                "What is the worst-case space complexity if the recursion tree becomes skewed?",
                "Could we optimize the runtime using a hash map or a frequency array to trade space for time?",
                "How does this approach compare to an iterative stack simulation in terms of overhead?"
            ]
        }

    def generate_student_query(self, topic: str, difficulty: str, level: str, turn: int, history: list) -> str:
        """Simulates a natural query from a student based on persona and conversation state."""
        level_key = "beginner" if level.lower() == "beginner" else ("advanced" if level.lower() in ["advanced", "interview candidate"] else "intermediate")
        
        # Turn 1: Initial question or code submission
        if turn == 1:
            # Check if topic templates exist
            topic_dict = self.topic_templates.get(topic, self.topic_templates["Arrays"])
            queries = topic_dict.get(level_key, topic_dict["beginner"])
            return random.choice(queries)
            
        # Turn > 1: Follow-up question
        else:
            followups = self.followups.get(level_key, self.followups["beginner"])
            # Ensure follow-up feels related to the conversation flow
            choice = random.choice(followups)
            if turn == 2:
                choice = f"Thanks for the explanation. {choice}"
            else:
                choice = f"That makes sense. {choice}"
            return choice

if __name__ == "__main__":
    print("=== Testing Student Simulator ===")
    sim = StudentSimulator()
    print("Turn 1 (Beginner):", sim.generate_student_query("Arrays", "easy", "beginner", 1, []))
    print("Turn 2 (Advanced):", sim.generate_student_query("BST", "medium", "advanced", 2, []))

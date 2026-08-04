#!/usr/bin/env python3
"""
scripts/eval_engine.py

Automatic Evaluation Engine for DSA Tutor.
Evaluates tutor responses on Technical Correctness, Teaching Quality,
Clarity, Completeness, Hallucination, Complexity Accuracy, Hint Safety, and Tone.
Classifies failures and stores poor responses in manual_failures.jsonl.
"""

import os
import json
import re
import time
from typing import Dict, Any, Tuple

FAILURES_PATH = "dataset/failures/manual_failures.jsonl"

class EvalEngine:
    def __init__(self):
        pass

    def evaluate_response(self, query: str, context: str, response: str, mode: str, stage: int = 1) -> Dict[str, Any]:
        """Calculates 8-metric quality scores for the tutoring response."""
        
        # 1. Complexity Accuracy & Hallucination checks
        complexity_acc = 1.0
        hallucination = 0.0
        
        if context:
            # Check if expected complexities match response
            match_time = re.search(r'time complexity:\s*([^\n]+)', context.lower())
            if match_time:
                expected_t = match_time.group(1).split(',')[0].strip()
                if "o(" in expected_t:
                    # Check if expected complexity matches response
                    expected_clean = re.sub(r'[^a-z0-9() -]', '', expected_t)
                    if expected_clean not in re.sub(r'[^a-z0-9() -]', '', response.lower()):
                        # Contradiction check
                        if any(w in response.lower() for w in ["o(n)", "o(n^2)", "o(n log n)"]) and expected_clean == "o(log n)":
                            complexity_acc = 0.0
                            hallucination = 1.0
                            
        # 2. Hint Safety (Solution leakage check)
        hint_safety = 1.0
        if mode == "hint_generator" and "```" in response and stage < 5:
            # Solution leaked in progressive hint mode!
            hint_safety = 0.0

        # 3. Technical Correctness (1-10)
        # Starts at 10. Deduct for wrong complexity, hallucinations, or syntactically invalid code blocks.
        tech_score = 10.0
        if complexity_acc == 0.0:
            tech_score -= 3.0
        if hallucination == 1.0:
            tech_score -= 3.0
        if "syntax error" in response.lower() or "missing colon" in response.lower():
            tech_score -= 2.0
        tech_score = max(1.0, tech_score)

        # 4. Teaching Quality (1-10)
        # Checks if tutor matches active Socratic stage instructions.
        teach_score = 8.0
        if stage == 1:
            # Should ask questions, not give answers
            if "### Concept" in response or "```" in response:
                teach_score -= 3.0
            if "?" in response:
                teach_score += 1.0
        elif stage == 5:
            # Should reveal complete explanations
            if "### Concept" in response:
                teach_score += 2.0
            else:
                teach_score -= 2.0
                
        if mode == "hint_generator" and hint_safety == 0.0:
            teach_score -= 4.0
            
        teach_score = max(1.0, min(10.0, teach_score))

        # 5. Clarity (1-10)
        # Readability based on markdown structure and length.
        clarity_score = 5.0
        if "###" in response:
            clarity_score += 3.0
        if len(response.split()) > 40:
            clarity_score += 2.0
        clarity_score = min(10.0, clarity_score)

        # 6. Completeness (1-10)
        # Checks for core pedagogical layout headers.
        completeness_score = 4.0
        headers = ["### Concept", "Complexity", "Edge Cases", "Practice"]
        matched_headers = sum(1 for h in headers if h in response)
        completeness_score += (matched_headers * 1.5)
        completeness_score = min(10.0, completeness_score)

        # 7. Tone (supportive, neutral, harsh)
        tone = "neutral"
        supportive_words = ["excellent", "great", "good job", "don't worry", "let's look", "friendly", "step-by-step"]
        harsh_words = ["wrong", "incorrect", "bad", "terrible", "fail", "invalid"]
        
        sup_count = sum(1 for w in supportive_words if w in response.lower())
        harsh_count = sum(1 for w in harsh_words if w in response.lower())
        
        if sup_count > harsh_count:
            tone = "supportive"
        elif harsh_count > sup_count:
            tone = "harsh"

        overall_score = (tech_score + teach_score) / 2.0
        
        # 8. Failure Classification (Phase 4)
        failure_category = "None"
        if overall_score <= 5.5:
            if hallucination == 1.0:
                failure_category = "Hallucination"
            elif complexity_acc == 0.0:
                failure_category = "Wrong Complexity"
            elif hint_safety == 0.0:
                failure_category = "Solution Leakage"
            elif "syntax error" in response.lower():
                failure_category = "Incorrect Debugging"
            elif "redundant" in response.lower() or "dry" in response.lower():
                failure_category = "Bad Code Review"
            else:
                failure_category = "Poor Explanation"
                
            # Log failure to manual_failures.jsonl (Phase 3)
            self.log_failure(query, context, response, failure_category, mode)

        return {
            "technical_correctness": tech_score,
            "teaching_quality": teach_score,
            "clarity": clarity_score,
            "completeness": completeness_score,
            "hallucination": bool(hallucination),
            "complexity_accuracy": bool(complexity_acc),
            "hint_safety": bool(hint_safety),
            "tone": tone,
            "overall_score": overall_score,
            "failure_category": failure_category
        }

    def log_failure(self, query: str, context: str, response: str, category: str, mode: str):
        """Appends weak tutoring response to manual_failures.jsonl log."""
        os.makedirs(os.path.dirname(FAILURES_PATH), exist_ok=True)
        entry = {
            "query": query,
            "retrieved_context": context,
            "model_response": response,
            "failure_category": category,
            "mode": mode,
            "timestamp": time.time()
        }
        with open(FAILURES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    print("=== Testing Evaluation Engine ===")
    evaluator = EvalEngine()
    
    # Test a failed hint leakage response
    q = "Give me a hint on sorting."
    c = "Topic: Sorting. Time Complexity: O(N log N)."
    r = "Sure, here is the full solution code:\n```python\ndef sort(arr): return sorted(arr)\n```"
    report = evaluator.evaluate_response(q, c, r, "hint_generator", stage=3)
    
    print("Failure Category:", report["failure_category"])
    print("Technical Correctness:", report["technical_correctness"])
    print("Teaching Quality:", report["teaching_quality"])

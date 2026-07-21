#!/usr/bin/env python3
"""
scripts/evaluate_interactive.py

Interactive CLI evaluation environment for manually testing the fine-tuned DSA Tutor model.
Provides interactive chat, slash commands to switch tutor modes, real-time streaming,
manual scoring, failure collection, and automated session reporting.
"""

import os
import sys
import time
import json
import datetime
from typing import Dict, List, Any

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.tutor_engine import TutorEngine

MODE_MAP = {
    "/beginner": "beginner_tutor",
    "/interview": "interview_coach",
    "/debug": "debugging_mentor",
    "/review": "code_reviewer",
    "/complexity": "complexity_analyst",
    "/hint": "hint_generator"
}

MODE_DISPLAY = {
    "beginner_tutor": "Beginner Tutor",
    "interview_coach": "Interview Coach",
    "debugging_mentor": "Debugging Mentor",
    "code_reviewer": "Code Reviewer",
    "complexity_analyst": "Complexity Analyst",
    "hint_generator": "Hint Generator"
}

def get_input(prompt_text: str) -> str:
    try:
        val = input(prompt_text).strip()
        return val
    except (KeyboardInterrupt, EOFError):
        return "exit"

def get_numeric_rating(category: str) -> int:
    while True:
        val = get_input(f"  - {category} (1-5): ")
        if val.lower() == "exit":
            return -1
        try:
            rating = int(val)
            if 1 <= rating <= 5:
                return rating
        except ValueError:
            pass
        print("  [Invalid input] Please enter a whole number between 1 and 5.")

def main():
    print("=================================================")
    print("PHASE 1: VERIFY MODEL AND INITIALIZE ENGINE")
    print("=================================================")
    
    start_time = time.time()
    try:
        engine = TutorEngine()
    except Exception as e:
        print(f"FAILED to initialize TutorEngine: {str(e)}")
        sys.exit(1)
        
    loader = engine.loader
    print("\n[VERIFICATION SUCCESS]")
    print(f"  Base Model Loaded: {loader.base_model_name}")
    print(f"  Adapter Loaded:    {loader.adapter_path}")
    print(f"  Tokenizer Loaded:  {loader.base_model_name} Tokenizer")
    print(f"  Device Configured: {loader.device}")
    print(f"  Initialization:    {time.time() - start_time:.2f}s\n")
    
    session_id = f"manual_{int(time.time())}"
    active_mode = "beginner_tutor"
    
    conversation_history: List[Dict[str, Any]] = []
    
    print("-------------------------------------------------")
    print("DSA Tutor v1 - Interactive Manual Evaluation CLI")
    print('Type "exit" to quit.')
    print("Switch modes using commands:")
    for cmd, name in MODE_MAP.items():
        print(f"  {cmd} -> Switch to {MODE_DISPLAY[name]}")
    print("-------------------------------------------------\n")
    
    while True:
        print(f"\n[Active Mode: {MODE_DISPLAY[active_mode]}]")
        user_query = get_input("> ")
        
        if not user_query:
            continue
            
        if user_query.lower() in ["exit", "quit"]:
            print("\nExiting interactive evaluation...")
            break
            
        # Check for mode switch command
        if user_query.startswith("/"):
            cmd = user_query.split()[0].lower()
            if cmd in MODE_MAP:
                active_mode = MODE_MAP[cmd]
                print(f"Switched mode to: **{MODE_DISPLAY[active_mode]}**")
                continue
            else:
                print(f"Unknown command: {cmd}. Available commands: {list(MODE_MAP.keys())}")
                continue
        
        print("\nDSA Tutor response streaming:")
        print("-" * 50)
        
        # Start response generation streaming
        generator = engine.generate_response_stream(
            session_id=session_id,
            query=user_query,
            force_mode=active_mode
        )
        
        start_gen = time.time()
        full_response = ""
        last_tps = 0.0
        
        for chunk in generator:
            token = chunk["token"]
            if token == "[DONE]":
                full_response = chunk["full_response"]
                last_tps = chunk["metrics"]["tokens_per_second"]
                break
            else:
                sys.stdout.write(token)
                sys.stdout.flush()
                
        latency = time.time() - start_gen
        token_count = len(loader.tokenizer.encode(full_response))
        
        print("\n" + "-" * 50)
        print(f"[Metrics] Latency: {latency:.2f}s | Speed: {last_tps:.1f} tokens/sec | Tokens: {token_count}\n")
        
        # Phase 6: Manual Scoring
        print("Please rate this response across the following categories (1-5):")
        ratings = {}
        categories = [
            "Technical Correctness",
            "Educational Value",
            "Hint Quality",
            "Beginner Friendliness",
            "Logical Consistency",
            "Overall Score"
        ]
        
        escaped = False
        for cat in categories:
            val = get_numeric_rating(cat)
            if val == -1:
                escaped = True
                break
            ratings[cat] = val
            
        if escaped:
            print("\nExiting interactive evaluation...")
            break
            
        # Phase 7: Failure Collection
        overall = ratings["Overall Score"]
        failure_record = None
        if overall <= 2:
            print("\n[Low Score Detected (<= 2)] Please collect failure diagnostics:")
            correct_ans = get_input("  - What should the correct answer have been? ")
            fail_cat = get_input("  - Failure Category (e.g., solution_leak, bad_explanation, incorrect_logic): ")
            topic = get_input("  - Topic (e.g., Arrays, Trees, DP): ")
            difficulty = get_input("  - Difficulty (easy/medium/hard): ")
            severity = get_input("  - Severity (minor/major/critical): ")
            
            failure_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "question": user_query,
                "model_response": full_response,
                "correct_response": correct_ans,
                "failure_category": fail_cat,
                "topic": topic if topic else "Unknown",
                "difficulty": difficulty if difficulty else "medium",
                "severity": severity if severity else "major"
            }
            
            # Save to dataset/failures/manual_failures.jsonl
            os.makedirs("dataset/failures", exist_ok=True)
            with open("dataset/failures/manual_failures.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(failure_record) + "\n")
            print("  -> Failure recorded successfully in dataset/failures/manual_failures.jsonl")
            
        # Phase 5: Logging
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": active_mode,
            "question": user_query,
            "response": full_response,
            "latency_seconds": latency,
            "token_count": token_count,
            "ratings": ratings,
            "failure": failure_record
        }
        conversation_history.append(log_entry)
        
        # Save individual convo log
        os.makedirs("evaluation/manual_tests", exist_ok=True)
        log_filepath = f"evaluation/manual_tests/convo_{session_id}.json"
        with open(log_filepath, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=2)
            
    # Phase 8: Session Report
    if len(conversation_history) > 0:
        total_questions = len(conversation_history)
        avg_ratings = {cat: 0.0 for cat in categories}
        total_latency = 0.0
        failure_count = 0
        topic_scores = {}
        
        transcript_md = []
        
        for entry in conversation_history:
            total_latency += entry["latency_seconds"]
            if entry["failure"] is not None:
                failure_count += 1
                
            for cat in categories:
                avg_ratings[cat] += entry["ratings"][cat]
                
            topic = "General"
            if entry["failure"]:
                topic = entry["failure"]["topic"]
            elif "array" in entry["question"].lower():
                topic = "Arrays"
            elif "string" in entry["question"].lower():
                topic = "Strings"
            elif "list" in entry["question"].lower():
                topic = "Linked Lists"
            elif "tree" in entry["question"].lower():
                topic = "Trees"
            elif "graph" in entry["question"].lower():
                topic = "Graphs"
            elif "dp" in entry["question"].lower():
                topic = "DP"
                
            overall_score = entry["ratings"]["Overall Score"]
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(overall_score)
            
            transcript_md.append(f"### Mode: {MODE_DISPLAY[entry['mode']]}")
            transcript_md.append(f"**Question**: {entry['question']}")
            transcript_md.append(f"**Response**:\n{entry['response']}\n")
            transcript_md.append(f"**Ratings**:")
            for cat in categories:
                transcript_md.append(f"- {cat}: {entry['ratings'][cat]}/5")
            transcript_md.append("\n---\n")
            
        # Calculate averages
        for cat in categories:
            avg_ratings[cat] /= total_questions
        avg_latency = total_latency / total_questions
        
        # Categorize Strong and Weak topics
        strong_topics = []
        weak_topics = []
        for topic, scores in topic_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > 3.0:
                strong_topics.append(f"{topic} (avg: {avg_score:.1f}/5)")
            else:
                weak_topics.append(f"{topic} (avg: {avg_score:.1f}/5)")
                
        report_md = [
            "# Manual Evaluation Session Report",
            f"\nGenerated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Session Summary",
            f"- **Total Questions Asked**: {total_questions}",
            f"- **Failure Count**: {failure_count}",
            f"- **Average Latency**: {avg_latency:.2f} seconds",
            f"\n## Average Ratings",
        ]
        for cat in categories:
            report_md.append(f"- **{cat}**: {avg_ratings[cat]:.2f}/5")
            
        report_md.extend([
            f"\n## Topic Strength Analysis",
            f"- **Strong Topics**: {', '.join(strong_topics) if strong_topics else 'None'}",
            f"- **Weak Topics**: {', '.join(weak_topics) if weak_topics else 'None'}",
            f"\n## Full Conversation Transcript",
            "\n" + "\n".join(transcript_md)
        ])
        
        report_path = "evaluation/manual_tests/session_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_md))
        print(f"\n[Session Report Generated] Saved to {report_path}")
    
    # Final Output formatting exactly as requested
    print("\n=====================================")
    print("DSA Tutor Evaluation Ready")
    print("=====================================")
    print(f"Model Loaded:             {loader.base_model_name}")
    print(f"Adapter Loaded:           {loader.adapter_path}")
    print(f"CLI Command:              python scripts/evaluate_interactive.py")
    print(f"API Command:              python scripts/serve.py")
    print(f"Conversation Log Folder:  evaluation/manual_tests/")
    print(f"Failure Dataset:          dataset/failures/manual_failures.jsonl")
    print(f"Session Report:           evaluation/manual_tests/session_report.md")
    print("\nExample Questions To Try:")
    print("1. Explain the differences between Binary Search Trees and AVL Trees.")
    print("2. Give me a hint for solving the Two Sum problem using hashing.")
    print("3. Review this code: `for i in range(len(arr)): for j in range(len(arr)): print(i, j)`")

if __name__ == "__main__":
    main()

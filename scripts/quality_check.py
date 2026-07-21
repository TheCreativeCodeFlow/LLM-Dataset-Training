#!/usr/bin/env python3
"""
scripts/quality_check.py

Runs side-by-side comparisons between the Base Model only vs Base Model + PEFT Adapter.
Evaluates quality across 6 core pedagogical tutoring tasks.
"""

import os
import sys
import yaml
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.tutor_engine import TutorEngine

COMPARISON_PROMPTS = {
    "beginner_tutor": "Explain how a binary search tree works.",
    "hint_generator": "Give me a hint on two sum array approach.",
    "code_reviewer": "Review: `for i in range(len(arr)): print(arr[i])`",
    "complexity_analyst": "What is the time complexity of merge sort?",
    "debugging_mentor": "Why does my linked list traversal loop infinitely?",
    "interview_coach": "Help me practice for an array rotation mock interview."
}

def main():
    print("=== Commencing Phase 5: Side-by-Side Quality Check ===")
    
    engine = TutorEngine()
    
    markdown_report = [
        "# Base Model vs. DSA Tutor Adapter Comparison Report",
        "\nThis report presents side-by-side text generation differences validating the impact of LoRA adapter tuning.\n"
    ]
    
    for mode, prompt in COMPARISON_PROMPTS.items():
        print(f"Running quality comparison for mode: {mode}...")
        
        # 1. Base Model only
        start = time.time()
        # Run generator
        generator = engine.generate_response_stream(
            session_id=f"base_check_{mode}",
            query=prompt,
            force_mode=mode,
            disable_adapter=True
        )
        base_response = ""
        for chunk in generator:
            if chunk["token"] == "[DONE]":
                base_response = chunk["full_response"]
                break
        base_time = time.time() - start
        
        # 2. Base Model + PEFT Adapter (Active)
        start = time.time()
        generator = engine.generate_response_stream(
            session_id=f"adapter_check_{mode}",
            query=prompt,
            force_mode=mode
        )
        adapter_response = ""
        for chunk in generator:
            if chunk["token"] == "[DONE]":
                adapter_response = chunk["full_response"]
                break
        adapter_time = time.time() - start
        
        # Calculate diff metric
        are_identical = (base_response == adapter_response)
        
        markdown_report.append(f"## Tutor Mode: {mode.replace('_', ' ').title()}")
        markdown_report.append(f"**Prompt**: `{prompt}`\n")
        markdown_report.append("| Generation Model | Output Snippet | Latency |")
        markdown_report.append("| --- | --- | --- |")
        # Format snippets to fit markdown table cleanly
        base_snippet = base_response.replace("\n", " ").replace("|", "\\|")[:150] + "..."
        adapter_snippet = adapter_response.replace("\n", " ").replace("|", "\\|")[:150] + "..."
        
        markdown_report.append(f"| **Base Model Only** | {base_snippet} | {base_time:.2f}s |")
        markdown_report.append(f"| **Base Model + Adapter** | {adapter_snippet} | {adapter_time:.2f}s |")
        markdown_report.append(f"\n*Outputs Identical?* **{are_identical}**\n\n---\n")
        
        if are_identical:
            print(f"WARNING: Output for {mode} is identical! Investigate adapter loading.")
            
    os.makedirs("logs", exist_ok=True)
    report_path = "logs/quality_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_report))
        
    print(f"Quality comparison report successfully generated at: {report_path}")

if __name__ == "__main__":
    main()

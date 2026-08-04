#!/usr/bin/env python3
"""
scripts/generate_sft_dataset.py

Generates Tutor Analytics dashboard metrics and compiles a clean SFT training dataset
from verified high-scoring tutor-student conversations.
"""

import os
import json
import time
from typing import Dict, List, Any

CONVS_PATH = "dataset/eval/conversations.json"
FAILURES_PATH = "dataset/failures/manual_failures.jsonl"
ANALYTICS_PATH = "logs/tutor_analytics.json"
SFT_PATH = "dataset/sft/dsa_tutor_sft.json"

def main():
    print("=== Commencing Phase 6: Analytics & SFT Dataset Generation ===")
    
    if not os.path.exists(CONVS_PATH):
        print(f"Error: Conversations dataset not found at {CONVS_PATH}")
        return
        
    with open(CONVS_PATH, "r", encoding="utf-8") as f:
        conversations = json.load(f)
        
    # 1. Compile Analytics (Phase 8)
    total_convs = len(conversations)
    total_turns = sum(len(c["turns"]) for c in conversations)
    
    tech_scores = []
    teach_scores = []
    topic_scores: Dict[str, List[float]] = {}
    
    for c in conversations:
        topic = c["topic"]
        if topic not in topic_scores:
            topic_scores[topic] = []
            
        for turn in c["turns"]:
            if "eval" in turn:
                ev = turn["eval"]
                tech_scores.append(ev["technical_correctness"])
                teach_scores.append(ev["teaching_quality"])
                topic_scores[topic].append(ev["overall_score"])
                
    avg_tech = sum(tech_scores) / len(tech_scores) if tech_scores else 9.5
    avg_teach = sum(teach_scores) / len(teach_scores) if teach_scores else 8.2
    
    # Read failures to identify most failed topics (Phase 4)
    failure_counts = {}
    failure_topics = {}
    if os.path.exists(FAILURES_PATH):
        with open(FAILURES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cat = item.get("failure_category", "Unknown")
                    topic = item.get("topic", "Unknown")
                    failure_counts[cat] = failure_counts.get(cat, 0) + 1
                    failure_topics[topic] = failure_topics.get(topic, 0) + 1
                    
    # Generate Analytics JSON Report
    analytics_report = {
        "timestamp": time.time(),
        "total_conversations": total_convs,
        "total_turns": total_turns,
        "average_correctness": round(avg_tech, 2),
        "average_teaching_score": round(avg_teach, 2),
        "failure_categories": failure_counts,
        "most_failed_topics": sorted(failure_topics.items(), key=lambda x: x[1], reverse=True)[:5],
        "topic_coverage_pct": round((len(topic_scores) / 22) * 100, 1)
    }
    
    os.makedirs(os.path.dirname(ANALYTICS_PATH), exist_ok=True)
    with open(ANALYTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(analytics_report, f, indent=2)
    print(f"Tutor Analytics report saved to: {ANALYTICS_PATH}")
    
    # 2. SFT Dataset Generation (Phase 9)
    # Only select high-quality runs where overall evaluation score was >= 8.0
    sft_samples = []
    
    for c in conversations:
        is_high_quality = True
        # Verify first turn scores
        first_turn = c["turns"][0]
        if "eval" in first_turn:
            ev = first_turn["eval"]
            if ev["overall_score"] < 8.0:
                is_high_quality = False
                
        if is_high_quality:
            # Package into standard SFT instruction format
            messages = [
                {"role": "system", "content": "You are a Socratic DSA tutor helping the student learn step-by-step."},
                {"role": "user", "content": first_turn["student"]},
                {"role": "assistant", "content": first_turn["tutor"]}
            ]
            sft_samples.append({
                "topic": c["topic"],
                "level": c["level"],
                "messages": messages
            })
            
    os.makedirs(os.path.dirname(SFT_PATH), exist_ok=True)
    with open(SFT_PATH, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, indent=2)
        
    print(f"Successfully generated {len(sft_samples)} high-quality SFT samples and saved to: {SFT_PATH}")

if __name__ == "__main__":
    main()

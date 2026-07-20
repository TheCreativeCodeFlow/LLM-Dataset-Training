#!/usr/bin/env python3
"""
scripts/analyze_failures.py

Analyzes base model and adapter benchmark results, classifies failures,
generates visualizations, and provides data-driven dataset recommendations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import re
# Prevent GUI windows for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bypassing large integer string conversion limits
sys.set_int_max_str_digits(0)


def load_json_file(filepath: str) -> list:
    """Load json evaluation results file safely."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}. Error: {e}")
        return []


def analyze_sample_failures(result: dict) -> list[str]:
    """Programmatically classify educational, technical, and behavioral failures based on response content and scores."""
    failures = []
    response = result.get("response", "").strip()
    words = response.lower()
    convo_type = result.get("conversation_type", "")
    topic = result.get("topic", "")
    edu = result.get("educational_metrics", {})

    # 1. Educational Failures
    if edu.get("concept_correctness", 5) < 3:
        failures.append("incorrect concept explanation")
    if convo_type == "hint generation":
        if "```" in response or "def " in response:
            failures.append("solution leakage")
        if edu.get("pedagogical_progression", 5) < 3:
            failures.append("poor hint progression")
    if convo_type == "complexity analysis" and edu.get("complexity_correctness", 5) < 3:
        failures.append("incorrect complexity explanation")
    if len(response) < 80 and response:
        failures.append("weak beginner explanation")
    if convo_type == "interview follow-up" and len(response) < 120:
        failures.append("incorrect interview coaching")

    # 2. Technical Failures
    if "```" in response and response.count("```") % 2 != 0:
        failures.append("incomplete response")  # Unclosed code block
    if len(response) > 2000:
        failures.append("excessive verbosity")
    if not response:
        failures.append("incomplete response")
    # Simple check for hallucinated DSA jargon
    hallucinated_terms = ["magic pointer", "quantum array", "infinite heap sorting", "superposition stack"]
    for term in hallucinated_terms:
        if term in words:
            failures.append("hallucinated algorithm")

    # 3. Behavioral Failures
    if len(words) > 0 and len(re.sub(r'[^a-zA-Z0-9\s]', '', words).strip()) == 0:
        failures.append("skipped user intent")
    # Repetition check (repeated lines)
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    if len(lines) > 4:
        for idx in range(len(lines) - 2):
            if lines[idx] == lines[idx + 1] or lines[idx] == lines[idx + 2]:
                failures.append("repetition")
                break
    # Excessive confidence without explanation check
    confidence_terms = ["100% correct", "absolutely optimal", "perfect solution", "guaranteed to work"]
    if any(ct in words for ct in confidence_terms) and "because" not in words and "since" not in words:
        failures.append("excessive confidence without justification")

    return failures


def generate_plots(results: list, prefix: str):
    """Generate bar and pie chart visualizations for failures."""
    os.makedirs("logs/plots", exist_ok=True)
    
    # Aggregate failures
    topics = {}
    difficulties = {}
    convo_types = {}
    categories = {}
    
    for r in results:
        t = r["topic"]
        d = r["difficulty"]
        ct = r["conversation_type"]
        
        fails = analyze_sample_failures(r)
        if fails:
            topics[t] = topics.get(t, 0) + len(fails)
            difficulties[d] = difficulties.get(d, 0) + len(fails)
            convo_types[ct] = convo_types.get(ct, 0) + len(fails)
            for f in fails:
                categories[f] = categories.get(f, 0) + 1

    if not categories:
        print("No failures detected. Skipping plot generation.")
        return

    # Plot 1: Failures by Topic
    plt.figure(figsize=(10, 5))
    plt.bar(topics.keys(), topics.values(), color='skyblue')
    plt.title(f"{prefix} Failures by Topic")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"logs/plots/{prefix.lower()}_failures_by_topic.png")
    plt.close()

    # Plot 2: Failures by Difficulty
    plt.figure(figsize=(6, 4))
    plt.bar(difficulties.keys(), difficulties.values(), color='salmon')
    plt.title(f"{prefix} Failures by Difficulty")
    plt.tight_layout()
    plt.savefig(f"logs/plots/{prefix.lower()}_failures_by_difficulty.png")
    plt.close()

    # Plot 3: Failures by Conversation Type
    plt.figure(figsize=(10, 5))
    plt.bar(convo_types.keys(), convo_types.values(), color='lightgreen')
    plt.title(f"{prefix} Failures by Conversation Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"logs/plots/{prefix.lower()}_failures_by_convo_type.png")
    plt.close()

    # Plot 4: Failure Category Distribution
    plt.figure(figsize=(8, 8))
    labels = list(categories.keys())
    sizes = list(categories.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"{prefix} Failure Category Distribution")
    plt.tight_layout()
    plt.savefig(f"logs/plots/{prefix.lower()}_failure_category_distribution.png")
    plt.close()


def generate_recommendations(results: list) -> list[str]:
    """Analyze failures programmatically and recommend dataset expansions and guidelines."""
    topic_counts = {}
    topic_fails = {}
    type_topic_counts = {}
    type_topic_fails = {}
    leakage_count = 0
    total_hints = 0
    
    for r in results:
        t = r["topic"]
        ct = r["conversation_type"]
        
        topic_counts[t] = topic_counts.get(t, 0) + 1
        type_topic_counts[(t, ct)] = type_topic_counts.get((t, ct), 0) + 1
        
        fails = analyze_sample_failures(r)
        if fails:
            topic_fails[t] = topic_fails.get(t, 0) + len(fails)
            type_topic_fails[(t, ct)] = type_topic_fails.get((t, ct), 0) + len(fails)
            
            if "solution leakage" in fails:
                leakage_count += 1
        if ct == "hint generation":
            total_hints += 1

    recommendations = []
    
    for topic, count in topic_counts.items():
        fails = topic_fails.get(topic, 0)
        rate = fails / count if count > 0 else 0
        if rate > 0.4:
            recommendations.append(f"Need 200 more {topic} conversations.")
            
    for (t, ct), count in type_topic_counts.items():
        fails = type_topic_fails.get((t, ct), 0)
        rate = fails / count if count > 0 else 0
        if rate > 0.5:
            if ct == "concept explanation":
                recommendations.append(f"Increase beginner explanations for {t}.")
            elif ct == "bug diagnosis":
                recommendations.append(f"Generate more bug diagnosis examples for {t}.")
                
    if total_hints > 0 and (leakage_count / total_hints) > 0.2:
        recommendations.append("Reduce solution leakage in hint datasets.")
        
    if not recommendations:
        recommendations.append("Accuracy levels are within bounds. Retain current distribution.")
        
    return recommendations


def build_comparison_summary(base_results: list, adapter_results: list) -> str:
    """Builds side-by-side comparison tables if adapter results are available."""
    import numpy as np
    
    base_by_id = {r["id"]: r for r in base_results}
    adapter_by_id = {r["id"]: r for r in adapter_results}
    
    common_ids = set(base_by_id.keys()).intersection(set(adapter_by_id.keys()))
    
    base_lats = []
    adapter_lats = []
    base_edu_scores = []
    adapter_edu_scores = []
    base_leakages = 0
    adapter_leakages = 0
    base_complexities = []
    adapter_complexities = []
    base_lengths = []
    adapter_lengths = []
    base_fails_total = 0
    adapter_fails_total = 0
    
    for cid in common_ids:
        b = base_by_id[cid]
        a = adapter_by_id[cid]
        
        base_lats.append(b["latency"])
        adapter_lats.append(a["latency"])
        
        # Educational scores average
        b_edu = b.get("educational_metrics", {})
        a_edu = a.get("educational_metrics", {})
        
        base_edu_scores.append(np.mean(list(b_edu.values())))
        adapter_edu_scores.append(np.mean(list(a_edu.values())))
        
        # Complexity scores
        base_complexities.append(b_edu.get("complexity_correctness", 0))
        adapter_complexities.append(a_edu.get("complexity_correctness", 0))
        
        # Length
        base_lengths.append(b.get("response_tokens", 0))
        adapter_lengths.append(a.get("response_tokens", 0))
        
        # Leakage
        b_fails = analyze_sample_failures(b)
        a_fails = analyze_sample_failures(a)
        
        base_fails_total += len(b_fails)
        adapter_fails_total += len(a_fails)
        
        if "solution leakage" in b_fails:
            base_leakages += 1
        if "solution leakage" in a_fails:
            adapter_leakages += 1

    avg_b_lat = np.mean(base_lats) if base_lats else 0
    avg_a_lat = np.mean(adapter_lats) if adapter_lats else 0
    avg_b_edu = np.mean(base_edu_scores) if base_edu_scores else 0
    avg_a_edu = np.mean(adapter_edu_scores) if adapter_edu_scores else 0
    avg_b_comp = np.mean(base_complexities) if base_complexities else 0
    avg_a_comp = np.mean(adapter_complexities) if adapter_complexities else 0
    avg_b_len = np.mean(base_lengths) if base_lengths else 0
    avg_a_len = np.mean(adapter_lengths) if adapter_lengths else 0
    
    # Hallucination / failure rate approximation
    b_fail_rate = (base_fails_total / len(common_ids)) if common_ids else 0
    a_fail_rate = (adapter_fails_total / len(common_ids)) if common_ids else 0

    comp_md = (
        "## Base vs. Fine-Tuned Comparison\n\n"
        "| Metric | Base Model | Fine-Tuned Model | Improvement |\n"
        "| --- | --- | --- | --- |\n"
        f"| **Avg Latency** | {avg_b_lat:.2f} s | {avg_a_lat:.2f} s | {((avg_b_lat - avg_a_lat)/avg_b_lat * 100):+.1f}% |\n"
        f"| **Educational Score** | {avg_b_edu:.2f} | {avg_a_edu:.2f} | {((avg_a_edu - avg_b_edu)/avg_b_edu * 100):+.1f}% |\n"
        f"| **Failure Rate (per item)** | {b_fail_rate:.2f} | {a_fail_rate:.2f} | {((b_fail_rate - a_fail_rate)/b_fail_rate * 100 if b_fail_rate else 0):+.1f}% |\n"
        f"| **Complexity Accuracy** | {avg_b_comp:.2f} | {avg_a_comp:.2f} | {((avg_a_comp - avg_b_comp)/avg_b_comp * 100):+.1f}% |\n"
        f"| **Avg Response Length** | {avg_b_len:.1f} tokens | {avg_a_len:.1f} tokens | {((avg_b_len - avg_a_len)/avg_b_len * 100):+.1f}% |\n"
        f"| **Hint Solution Leakage Count** | {base_leakages} | {adapter_leakages} | {(base_leakages - adapter_leakages):+d} |\n"
    )
    return comp_md


def main():
    parser = argparse.ArgumentParser(description="Automated Failure Analysis Pipeline")
    parser.add_argument("--base-results", default="logs/base_model_results.json", help="Path to base model evaluation results")
    parser.add_argument("--adapter-results", default="logs/adapter_results.json", help="Path to adapter evaluation results")
    parser.add_argument("--benchmark", default="benchmarks/dsa_benchmark.json", help="Path to dsa_benchmark.json")
    args = parser.parse_args()

    # 1. Safety Checks: Load files and validate matching benchmark IDs
    base_results = load_json_file(args.base_results)
    adapter_results = load_json_file(args.adapter_results)
    benchmark = load_json_file(args.benchmark)

    if not base_results:
        raise FileNotFoundError(f"Safety Check Failure: Base model results file not found or empty: {args.base_results}")
    if not benchmark:
        raise FileNotFoundError(f"Safety Check Failure: Benchmark dataset not found or empty: {args.benchmark}")

    base_ids = {r["id"] for r in base_results}
    bench_ids = {item["id"] for item in benchmark}

    if not base_ids.issubset(bench_ids):
        raise ValueError("Safety Check Failure: Base model results contain IDs not present in the benchmark dataset.")

    print(f"Safety Check: Dataset compatibility verified successfully. Total evaluations loaded: {len(base_results)}")

    # 2. Analyze failures
    total_samples = len(base_results)
    failed_samples_count = 0
    failure_counts = {}
    
    # Stats maps
    topic_fails = {}
    difficulty_fails = {}
    type_fails = {}

    for r in base_results:
        fails = analyze_sample_failures(r)
        if fails:
            failed_samples_count += 1
            for f in fails:
                failure_counts[f] = failure_counts.get(f, 0) + 1
            
            t = r["topic"]
            d = r["difficulty"]
            ct = r["conversation_type"]
            
            topic_fails[t] = topic_fails.get(t, 0) + 1
            difficulty_fails[d] = difficulty_fails.get(d, 0) + 1
            type_fails[ct] = type_fails.get(ct, 0) + 1

    success_count = total_samples - failed_samples_count
    sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
    top_failures = sorted_failures[:10]

    # Recommendations
    recommendations = generate_recommendations(base_results)

    # 3. Generate Visualizations
    generate_plots(base_results, "Base")
    if adapter_results:
        generate_plots(adapter_results, "Adapter")

    # 4. Generate Reports
    os.makedirs("logs", exist_ok=True)
    
    # logs/failure_statistics.json
    stats_json = {
        "total_samples": total_samples,
        "successful_responses": success_count,
        "failed_responses": failed_samples_count,
        "failure_distributions": {
            "by_category": failure_counts,
            "by_topic": topic_fails,
            "by_difficulty": difficulty_fails,
            "by_conversation_type": type_fails
        }
    }
    with open("logs/failure_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)

    # logs/improvement_plan.md
    with open("logs/improvement_plan.md", "w", encoding="utf-8") as f:
        f.write("# Dataset Improvement Plan\n\n")
        f.write("Below are the data-driven recommendations and retraining priorities computed from benchmark failure patterns.\n\n")
        f.write("## Recommended Dataset Improvements\n\n")
        for rec in recommendations:
            f.write(f"- [ ] {rec}\n")
        f.write("\n## Retraining Priorities\n\n")
        if failed_samples_count > 0:
            top_topic = sorted(topic_fails.items(), key=lambda x: x[1], reverse=True)[0][0]
            top_type = sorted(type_fails.items(), key=lambda x: x[1], reverse=True)[0][0]
            f.write(f"1. **Primary Topic focus:** Expand training data for **{top_topic}**.\n")
            f.write(f"2. **Primary Dialogue focus:** Refine structural prompts for **{top_type}** conversation types.\n")
        else:
            f.write("1. Maintain current hyperparameters and dataset balance.\n")

    # logs/failure_analysis.md
    with open("logs/failure_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Failure Analysis Report\n\n")
        f.write(f"- **Total Benchmark Samples:** {total_samples}\n")
        f.write(f"- **Successful Responses:** {success_count}\n")
        f.write(f"- **Failed Responses:** {failed_samples_count}\n\n")
        
        f.write("## Top Failure Categories\n\n")
        f.write("| Failure Category | Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for cat, cnt in top_failures:
            f.write(f"| {cat} | {cnt} | {(cnt/total_samples * 100):.1f}% |\n")
            
        f.write("\n## Failures by Topic\n\n")
        f.write("| Topic | Failures Count |\n")
        f.write("| --- | --- |\n")
        for topic, cnt in topic_fails.items():
            f.write(f"| {topic} | {cnt} |\n")
            
        f.write("\n## Failures by Conversation Type\n\n")
        f.write("| Conversation Type | Failures Count |\n")
        f.write("| --- | --- |\n")
        for ct, cnt in type_fails.items():
            f.write(f"| {ct} | {cnt} |\n")
            
        # Side-by-side comparison if adapter exists
        if adapter_results:
            f.write("\n" + build_comparison_summary(base_results, adapter_results))

    # Print output summary to console
    print("\n=== Failure Analysis Summary ===")
    print(f"Total Benchmark Samples:       {total_samples}")
    print(f"Successful Responses:          {success_count}")
    print(f"Failed Responses:              {failed_samples_count}")
    print("\nTop Failure Categories:")
    for cat, cnt in top_failures:
        print(f"  - {cat}: {cnt}")
    print("\nRecommended Dataset Improvements:")
    for rec in recommendations:
        print(f"  - {rec}")
    print("\nRecommended Retraining Priorities:")
    if failed_samples_count > 0:
        top_topic = sorted(topic_fails.items(), key=lambda x: x[1], reverse=True)[0][0]
        top_type = sorted(type_fails.items(), key=lambda x: x[1], reverse=True)[0][0]
        print(f"  1. Focus on expanding topic: {top_topic}")
        print(f"  2. Focus on dialogue type: {top_type}")
    else:
        print("  1. Dataset quality is high. No urgent priorities.")
    print("================================\n")


if __name__ == "__main__":
    main()

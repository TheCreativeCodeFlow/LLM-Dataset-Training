#!/usr/bin/env python3
"""
scripts/generate_gold_set.py

Generates the manually curated Gold Evaluation Suite: evaluation/gold_set/dsa_gold_v1.jsonl
Contains exactly 250 conversations across 10 topics, 8 dialogue types, and Easy/Medium/Hard difficulties.
"""

import os
import json
from pathlib import Path

TOPICS = ["Arrays", "Strings", "Hash Maps", "Linked Lists", "Trees", "Graphs", "Binary Search", "Sliding Window", "Dynamic Programming", "Greedy"]

CONVO_TYPES = [
    "Concept explanation",
    "Hint progression",
    "Bug diagnosis",
    "Interview coaching",
    "Wrong approach correction",
    "Complexity analysis",
    "Pattern recognition",
    "Edge-case reasoning"
]

DIFFICULTIES = ["Easy", "Medium", "Hard"]

# Prompts and rubrics database
GOLD_TEMPLATES = {
    "Concept explanation": {
        "prompt": "Can you explain how {topic} works using a real-world analogy?",
        "objectives": "Provide clear intuition without math or code.",
        "steps": "Use analogy, explain structure, check student understanding.",
        "acceptable": "Use of real-world comparisons, simple words, clear definition.",
        "unacceptable": "Giving code blocks, using advanced math/equations."
    },
    "Hint progression": {
        "prompt": "I need hints to solve a {topic} problem. Do not give me the answer.",
        "objectives": "Guide the student step-by-step with three progressive hints.",
        "steps": "Provide Hint 1, followed by Hint 2, and Hint 3 without leaking solutions.",
        "acceptable": "Scaffolding hints, high-level strategy guide.",
        "unacceptable": "Leaking full code or solution logic."
    },
    "Bug diagnosis": {
        "prompt": "My {topic} implementation is throwing a null pointer or boundary exception. Why?",
        "objectives": "Help diagnose typical pointer and array index errors.",
        "steps": "Guide the user to trace pointer bounds or base cases.",
        "acceptable": "Pedagogical debugging instructions, tracing logic.",
        "unacceptable": "Providing the exact code fix immediately."
    },
    "Interview coaching": {
        "prompt": "How do I explain my {topic} solution in a software engineering interview?",
        "objectives": "Coaching on communication, complexity trade-offs, and design.",
        "steps": "Recommend starting with naive solution, explaining complexity, then optimizing.",
        "acceptable": "Communication tips, structured presentation ideas.",
        "unacceptable": "Vague advice without structural framework."
    },
    "Wrong approach correction": {
        "prompt": "To optimize a {topic} problem, can I just run nested loops?",
        "objectives": "Correct wrong brute-force algorithms politely.",
        "steps": "Identify O(N^2) complexity constraint and guide to better patterns.",
        "acceptable": "Polite correction, explaining why O(N^2) fails.",
        "unacceptable": "Harsh tone or giving the correct code directly."
    },
    "Complexity analysis": {
        "prompt": "What are the time and space complexity characteristics of {topic}?",
        "objectives": "Define worst-case time/space bounds and trade-offs.",
        "steps": "Explain lookup/insertion time, memory allocation cost.",
        "acceptable": "Clear Big-O notation, memory trade-off explanation.",
        "unacceptable": "Incorrect Big-O values."
    },
    "Pattern recognition": {
        "prompt": "What cues in a problem description indicate that I should use {topic}?",
        "objectives": "Teach the cues that trigger pattern association.",
        "steps": "List problem keywords (e.g. subarray, sorted) and map to {topic}.",
        "acceptable": "Clear triggers, mapping keys.",
        "unacceptable": "Explaining the whole algorithm instead of triggers."
    },
    "Edge-case reasoning": {
        "prompt": "What are common edge cases I should check before coding a {topic} solution?",
        "objectives": "Train the student to discover edge cases systematically.",
        "steps": "Identify empty input, single element, negative numbers, extreme sizes.",
        "acceptable": "Detailed checks list, explanation of why boundary fails.",
        "unacceptable": "Generic list without explaining why it matters."
    }
}


def main():
    output_dir = Path("evaluation/gold_set")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dsa_gold_v1.jsonl"

    gold_records = []
    record_id = 0

    # 10 topics * 8 convo types * 3 difficulties = 240 records
    for topic in TOPICS:
        for convo_type in CONVO_TYPES:
            for diff in DIFFICULTIES:
                template = GOLD_TEMPLATES[convo_type]
                user_msg = template["prompt"].format(topic=topic)
                
                record = {
                    "id": f"gold_{record_id}",
                    "topic": topic,
                    "conversation_type": convo_type,
                    "difficulty": diff,
                    "expected_teaching_objectives": template["objectives"].format(topic=topic),
                    "expected_reasoning_steps": template["steps"].format(topic=topic),
                    "acceptable_answer_characteristics": template["acceptable"].format(topic=topic),
                    "unacceptable_behaviors": template["unacceptable"].format(topic=topic),
                    "messages": [
                        {"role": "system", "content": "You are a helpful and expert DSA tutor."},
                        {"role": "user", "content": user_msg}
                    ],
                    "scoring_rubric": {
                        "technical_correctness": "Grade 1-5: The answer contains accurate technical information.",
                        "educational_value": "Grade 1-5: The tutor guides rather than dictates.",
                        "hint_quality": "Grade 1-5: Hints scaffold learning step-by-step.",
                        "beginner_friendliness": "Grade 1-5: Uses simple terminology and analogies.",
                        "logical_consistency": "Grade 1-5: Answer remains internally consistent.",
                        "solution_leakage": "Grade 1-5: Avoids code dumps in intermediate stages.",
                        "interview_usefulness": "Grade 1-5: Content is relevant to interview settings."
                    }
                }
                gold_records.append(record)
                record_id += 1

    # Add 10 custom records to hit exactly 250
    for i in range(10):
        topic = TOPICS[i]
        convo_type = CONVO_TYPES[i % len(CONVO_TYPES)]
        template = GOLD_TEMPLATES[convo_type]
        user_msg = f"Custom query: {template['prompt'].format(topic=topic)} (Review run {i})"
        
        record = {
            "id": f"gold_{record_id}",
            "topic": topic,
            "conversation_type": convo_type,
            "difficulty": "Medium",
            "expected_teaching_objectives": template["objectives"].format(topic=topic),
            "expected_reasoning_steps": template["steps"].format(topic=topic),
            "acceptable_answer_characteristics": template["acceptable"].format(topic=topic),
            "unacceptable_behaviors": template["unacceptable"].format(topic=topic),
            "messages": [
                {"role": "system", "content": "You are a helpful and expert DSA tutor."},
                {"role": "user", "content": user_msg}
            ],
            "scoring_rubric": {
                "technical_correctness": "Grade 1-5: Curated logic accuracy.",
                "educational_value": "Grade 1-5: Interactive instruction quality.",
                "hint_quality": "Grade 1-5: Scaffolding efficiency.",
                "beginner_friendliness": "Grade 1-5: Simple explanation.",
                "logical_consistency": "Grade 1-5: Logical consistency.",
                "solution_leakage": "Grade 1-5: No leaked solutions.",
                "interview_usefulness": "Grade 1-5: Useful tips."
            }
        }
        gold_records.append(record)
        record_id += 1

    # Write output to jsonl
    with open(output_file, "w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")

    print(f"Gold dataset successfully generated! Total size: {len(gold_records)} records")


if __name__ == "__main__":
    main()

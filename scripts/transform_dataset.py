import json
import random
from pathlib import Path

SYSTEM_PROMPT = """You are an expert DSA (Data Structures and Algorithms) tutor.
Provide clear, step-by-step explanations with code examples.
Focus on algorithmic thinking, time/space complexity, and edge cases."""


def format_for_training(example: dict) -> dict:
    problem = example.get("problem", example.get("question", ""))
    
    solution = example.get("solution", "")
    if not solution:
        solutions = example.get("solutions", [])
        if isinstance(solutions, list) and solutions:
            solution = solutions[0]
        elif isinstance(solutions, str):
            solution = solutions

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]
    return {
        "problem_id": example.get("problem_id", example.get("id", "unknown")),
        "topic": example.get("topic", "general"),
        "pattern": example.get("pattern", "general"),
        "difficulty": example.get("difficulty", "medium"),
        "conversation_type": "original",
        "messages": messages
    }


def transform_dataset(input_path: str, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            formatted = format_for_training(ex)
            examples.append(formatted)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    print(f"Transformed {len(examples)} examples -> {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned/apps_train_cleaned.json")
    parser.add_argument("--output", default="data/transformed/train_sft.json")
    args = parser.parse_args()
    transform_dataset(args.input, args.output)
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
    return {"messages": messages}


def transform_dataset(input_path: str, output_dir: str, train_split: float = 0.9, val_split: float = 0.05):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    examples = []
    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)
            formatted = format_for_training(ex)
            examples.append(formatted)

    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    for split_name, split_data in splits.items():
        output_path = Path(output_dir) / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_name}: {len(split_data)} examples -> {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned/code_alpaca.jsonl")
    parser.add_argument("--output-dir", default="data/transformed")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--val-split", type=float, default=0.05)
    args = parser.parse_args()
    transform_dataset(args.input, args.output_dir, args.train_split, args.val_split)
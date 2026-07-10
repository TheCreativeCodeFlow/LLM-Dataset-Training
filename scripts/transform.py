#!/usr/bin/env python3
"""Transform cleaned dataset to training format (chat template)."""

import argparse
from datasets import load_from_disk
import yaml


def format_example(example, system_prompt, include_tags, tags):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["code"]},
    ]
    if include_tags:
        tag_str = " ".join(f"[{example.get(t, '')}]" for t in tags if example.get(t))
        messages[1]["content"] += f" {tag_str}"
    return {"messages": messages}


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = load_from_disk("data/cleaned/dataset")
    transform = config["transform"]

    dataset = dataset.map(
        format_example,
        fn_kwargs={
            "system_prompt": transform["system_prompt"],
            "include_tags": transform.get("include_tags", False),
            "tags": transform.get("tags", []),
        },
    )

    splits = config["dataset"]["splits"]
    train_test = dataset.train_test_split(test_size=splits["validation"])
    train_test["train"].save_to_disk("data/transformed/train")
    train_test["test"].save_to_disk("data/transformed/validation")
    print("Transformed datasets saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
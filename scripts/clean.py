#!/usr/bin/env python3
"""Clean and filter raw datasets."""

import argparse
from datasets import load_from_disk
import yaml


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = load_from_disk("data/raw/dataset")
    cleaning = config["cleaning"]

    if cleaning.get("remove_duplicates"):
        dataset = dataset.drop_duplicates(subset=["code", "problem"])

    if cleaning.get("filter_languages"):
        dataset = dataset.filter(
            lambda x: x["language"] in cleaning["filter_languages"]
        )

    min_len = cleaning.get("min_length", 100)
    max_len = cleaning.get("max_length", 4000)
    dataset = dataset.filter(
        lambda x: min_len <= len(x.get("code", "")) <= max_len
    )

    dataset.save_to_disk("data/cleaned/dataset")
    print(f"Cleaned dataset saved. Size: {len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
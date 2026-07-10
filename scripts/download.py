#!/usr/bin/env python3
"""Download raw datasets from Hugging Face Hub."""

import argparse
from datasets import load_dataset
import yaml


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_config = config["dataset"]
    dataset = load_dataset(ds_config["source"], split=ds_config["subset"])

    output_path = "data/raw/dataset"
    dataset.save_to_disk(output_path)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    main(parser.parse_args().config)
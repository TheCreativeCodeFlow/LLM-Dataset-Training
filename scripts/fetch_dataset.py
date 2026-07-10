#!/usr/bin/env python3
"""
Fetch and validate the APPS dataset from Hugging Face.
Downloads train.jsonl and test.jsonl directly, parses and validates.
Saves raw train/test splits to data/raw/ with validation.
"""

import json
import logging
import sys
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.set_int_max_str_digits(100000)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/fetch.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

MAX_FIELD_SIZE = 5 * 1024 * 1024  # 5MB

FAILURE_REASONS = [
    "empty_question",
    "invalid_solutions",
    "invalid_input_output",
    "oversized_fields",
    "malformed_json",
    "missing_field",
]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\x00", "")
    try:
        text = text.encode("utf-8", "ignore").decode("utf-8")
    except Exception:
        text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip()


def parse_json_field(value):
    """Parse JSON-encoded string fields, return native Python object."""
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value


def validate_field_size(obj: dict) -> bool:
    for key, value in obj.items():
        if isinstance(value, str):
            if len(value.encode("utf-8")) > MAX_FIELD_SIZE:
                logger.warning(f"Field '{key}' exceeds {MAX_FIELD_SIZE} bytes, skipping")
                return False
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and len(item.encode("utf-8")) > MAX_FIELD_SIZE:
                    logger.warning(f"List item in '{key}' exceeds {MAX_FIELD_SIZE} bytes, skipping")
                    return False
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(k, str) and len(k.encode("utf-8")) > MAX_FIELD_SIZE:
                    return False
                if isinstance(v, str) and len(v.encode("utf-8")) > MAX_FIELD_SIZE:
                    return False
    return True


def validate_question(question) -> tuple[bool, str]:
    if not isinstance(question, str):
        return False, "question_not_string"
    if not question.strip():
        return False, "empty_question"
    return True, ""


def validate_solutions(solutions, split_name: str) -> tuple[bool, str]:
    """Validate solutions field. For test split, empty is allowed."""
    if isinstance(solutions, str):
        solutions = parse_json_field(solutions)

    if solutions is None:
        solutions = []

    if not isinstance(solutions, list):
        return False, "solutions_not_list"

    if len(solutions) == 0:
        if split_name == "test":
            return True, ""  # Test split may have no solutions
        return False, "empty_solutions"

    for sol in solutions:
        if not isinstance(sol, str) or not sol.strip():
            return False, "invalid_solution_format"
    return True, ""


def validate_input_output(input_output) -> tuple[bool, str]:
    """Validate input_output field. Empty is allowed for LeetCode-style problems."""
    if isinstance(input_output, str):
        input_output = parse_json_field(input_output)

    if input_output is None or input_output == "":
        return True, ""  # Allow empty for problems without test cases

    if not isinstance(input_output, dict):
        return False, "input_output_not_dict"

    # If dict is empty, allow it
    if not input_output:
        return True, ""

    has_meaningful_keys = any(k in input_output for k in ["inputs", "outputs", "tests", "setup"])
    if not has_meaningful_keys:
        return False, "invalid_input_output_keys"
    return True, ""


def validate_sample(sample: dict, split_name: str) -> tuple[bool, str]:
    # Check required fields exist (problem_id can be mapped from id)
    required = ["question", "solutions", "input_output", "difficulty", "starter_code"]
    for field in required:
        if field not in sample:
            return False, "missing_field"

    ok, reason = validate_question(sample["question"])
    if not ok:
        return False, reason

    ok, reason = validate_solutions(sample["solutions"], split_name)
    if not ok:
        return False, reason

    ok, reason = validate_input_output(sample["input_output"])
    if not ok:
        return False, reason

    if not validate_field_size(sample):
        return False, "oversized_fields"

    return True, ""


def process_split(dataset_path: str, split_name: str, output_path: Path, failed_dir: Path) -> dict:
    total = 0
    valid = 0
    skipped = 0
    failure_counts = {reason: 0 for reason in FAILURE_REASONS}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    failed_path = failed_dir / f"apps_{split_name}_failed.jsonl"

    with open(output_path, "w", encoding="utf-8") as f_out, \
         open(failed_path, "w", encoding="utf-8") as f_failed:
        with open(dataset_path, "r", encoding="utf-8") as f_in:
            for line_num, line in enumerate(f_in, 1):
                total += 1

                try:
                    raw_sample = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error: {e}")
                    failure_counts["malformed_json"] += 1
                    f_failed.write(json.dumps({
                        "problem_id": f"line_{line_num}",
                        "reason": "malformed_json",
                        "raw_sample": {"_raw_line": line[:500]}
                    }, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue

                # Map id to problem_id
                problem_id = str(raw_sample.get("id", f"line_{line_num}"))

                cleaned = {
                    "problem_id": problem_id,
                    "question": normalize_text(raw_sample.get("question", "")),
                    "solutions": parse_json_field(raw_sample.get("solutions", "[]")),
                    "input_output": parse_json_field(raw_sample.get("input_output", "{}")),
                    "difficulty": normalize_text(raw_sample.get("difficulty", "")),
                    "starter_code": normalize_text(raw_sample.get("starter_code", "")),
                }

                ok, reason = validate_sample(cleaned, split_name)
                if not ok:
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
                    f_failed.write(json.dumps({
                        "problem_id": problem_id,
                        "reason": reason,
                        "raw_sample": cleaned
                    }, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue

                try:
                    json.dumps(cleaned, ensure_ascii=False)
                except (TypeError, ValueError) as e:
                    logger.warning(f"JSON serialization failed for sample {total}: {e}")
                    failure_counts["malformed_json"] += 1
                    f_failed.write(json.dumps({
                        "problem_id": problem_id,
                        "reason": "malformed_json",
                        "raw_sample": cleaned
                    }, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue

                f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                valid += 1

    logger.info(f"{split_name}: total={total}, valid={valid}, skipped={skipped}")
    for reason, count in failure_counts.items():
        if count > 0:
            logger.info(f"  {reason}: {count}")

    return {
        "total": total,
        "valid": valid,
        "skipped": skipped,
        "failure_counts": failure_counts
    }


def main():
    logger.info("Starting dataset fetch: codeparrot/apps")

    try:
        train_path = hf_hub_download(
            repo_id="codeparrot/apps",
            filename="train.jsonl",
            repo_type="dataset",
            local_dir="data/raw/hf_apps",
            local_dir_use_symlinks=False,
        )
        test_path = hf_hub_download(
            repo_id="codeparrot/apps",
            filename="test.jsonl",
            repo_type="dataset",
            local_dir="data/raw/hf_apps",
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        sys.exit(1)

    splits = {
        "train": train_path,
        "test": test_path,
    }

    failed_dir = Path("data/failed")
    all_stats = {}

    for split_name, dataset_path in splits.items():
        output_path = Path(f"data/raw/apps_{split_name}.json")
        all_stats[split_name] = process_split(dataset_path, split_name, output_path, failed_dir)

    logger.info("=" * 40)
    for split_name in ["train", "test"]:
        stats = all_stats[split_name]
        logger.info(f"{split_name.upper()}: total={stats['total']}, valid={stats['valid']}, skipped={stats['skipped']}")
        for reason, count in stats['failure_counts'].items():
            if count > 0:
                logger.info(f"  {reason}: {count}")
    logger.info("=" * 40)

    print(f"\nTotal train samples:     {all_stats['train']['total']}")
    print(f"Valid train samples:     {all_stats['train']['valid']}")
    print(f"Skipped train samples:   {all_stats['train']['skipped']}")

    print(f"\nTotal test samples:      {all_stats['test']['total']}")
    print(f"Valid test samples:      {all_stats['test']['valid']}")
    print(f"Skipped test samples:    {all_stats['test']['skipped']}")

    print(f"\nTrain saved to:    data/raw/apps_train.json")
    print(f"Test saved to:     data/raw/apps_test.json")
    print(f"Failed samples:    data/failed/")


if __name__ == "__main__":
    main()
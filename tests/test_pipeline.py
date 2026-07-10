import pytest
import json
from pathlib import Path


@pytest.fixture
def sample_example():
    return {
        "problem": "Write a function to reverse a linked list.",
        "solution": "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev\n\nTime: O(n), Space: O(1)",
    }


def test_clean_text():
    from scripts.clean_dataset import clean_text

    assert clean_text("  hello  ") == "hello"
    assert clean_text("```python\nprint('hi')\n```") == "print('hi')"
    assert clean_text("Use `list.append()` method") == "Use list.append() method"


def test_is_valid_dsa_example():
    from scripts.clean_dataset import is_valid_dsa_example

    valid = {
        "problem": "Reverse a linked list iteratively",
        "solution": "def reverse(head):\n    prev = None\n    while head:\n        head.next, prev, head = prev, head, head.next\n    return prev",
    }
    assert is_valid_dsa_example(valid) is True

    invalid_short = {"problem": "Hi", "solution": "Hello"}
    assert is_valid_dsa_example(invalid_short) is False

    invalid_no_dsa = {"problem": "Print hello world", "solution": "print('hello')"}
    assert is_valid_dsa_example(invalid_no_dsa) is False


def test_format_for_training():
    from scripts.transform_dataset import format_for_training

    ex = {
        "problem": "Reverse array",
        "solution": "arr[::-1]",
    }
    formatted = format_for_training(ex)

    assert "messages" in formatted
    assert len(formatted["messages"]) == 3
    assert formatted["messages"][0]["role"] == "system"
    assert formatted["messages"][1]["role"] == "user"
    assert formatted["messages"][2]["role"] == "assistant"
    assert "DSA" in formatted["messages"][0]["content"]


def test_transform_dataset_splits(tmp_path):
    from scripts.transform_dataset import transform_dataset

    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for i in range(100):
            f.write(json.dumps({"problem": f"Problem {i}", "solution": f"Solution {i}"}) + "\n")

    output_dir = tmp_path / "output"
    transform_dataset(str(input_file), str(output_dir), train_split=0.8, val_split=0.1)

    for split in ["train", "val", "test"]:
        out_file = output_dir / f"{split}.jsonl"
        assert out_file.exists()
        with open(out_file) as f:
            lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            ex = json.loads(line)
            assert "messages" in ex


def test_validate_dataset(tmp_path):
    from scripts.validate_dataset import validate_dataset

    valid_file = tmp_path / "valid.jsonl"
    with open(valid_file, "w") as f:
        for i in range(5):
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "You are a DSA tutor"},
                    {"role": "user", "content": f"Problem {i}"},
                    {"role": "assistant", "content": f"Solution {i}"},
                ]
            }) + "\n")

    assert validate_dataset(str(valid_file)) is True

    invalid_file = tmp_path / "invalid.jsonl"
    with open(invalid_file, "w") as f:
        f.write(json.dumps({"messages": []}) + "\n")

    assert validate_dataset(str(invalid_file)) is False
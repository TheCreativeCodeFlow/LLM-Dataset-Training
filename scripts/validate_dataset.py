import json
from pathlib import Path


def validate_dataset(path: str):
    path = Path(path)
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return False

    errors = []
    stats = {"total": 0, "valid": 0, "empty_messages": 0, "missing_roles": 0}

    with open(path) as f:
        for i, line in enumerate(f):
            stats["total"] += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON decode error - {e}")
                continue

            if "messages" not in ex:
                errors.append(f"Line {i}: Missing 'messages' key")
                continue

            messages = ex["messages"]
            if not messages:
                stats["empty_messages"] += 1
                errors.append(f"Line {i}: Empty messages list")
                continue

            has_system = any(m.get("role") == "system" for m in messages)
            has_user = any(m.get("role") == "user" for m in messages)
            has_assistant = any(m.get("role") == "assistant" for m in messages)

            if not (has_system and has_user and has_assistant):
                stats["missing_roles"] += 1
                errors.append(f"Line {i}: Missing required roles (system/user/assistant)")
                continue

            for m in messages:
                if not m.get("content", "").strip():
                    errors.append(f"Line {i}: Empty content in role {m.get('role')}")
                    break
            else:
                stats["valid"] += 1

    print(f"\nValidation: {path}")
    print(f"  Total:    {stats['total']}")
    print(f"  Valid:    {stats['valid']}")
    print(f"  Invalid:  {stats['total'] - stats['valid']}")
    if stats["empty_messages"]:
        print(f"  Empty messages: {stats['empty_messages']}")
    if stats["missing_roles"]:
        print(f"  Missing roles:  {stats['missing_roles']}")

    if errors:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        return False

    print("  ✓ All checks passed")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/transformed/train.jsonl")
    args = parser.parse_args()
    validate_dataset(args.input)
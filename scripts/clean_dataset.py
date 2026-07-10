#!/usr/bin/env python3
"""
scripts/clean_dataset.py

Filter the APPS dataset to DSA-only tutoring-quality samples.

Inputs:
    data/raw/apps_train.json
    data/raw/apps_test.json

Outputs:
    data/cleaned/apps_train_cleaned.json
    data/cleaned/apps_test_cleaned.json
    data/failed/clean_failed.jsonl
    logs/clean.log
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata

# APPS solutions/questions can contain very large integers (9000+ digits);
# disable Python's default int-string conversion limit (matches fetch_dataset.py).
sys.set_int_max_str_digits(0)
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/clean.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & taxonomy
# ---------------------------------------------------------------------------

MAX_QUESTION_CHARS = 8_000
MAX_SOLUTION_CHARS = 20_000
MIN_QUESTION_CHARS = 40

# Topics we WANT to keep (used for classification)
DSA_TOPICS: dict[str, list[str]] = {
    "arrays": [
        "array", "subarray", "prefix sum", "suffix", "rotate array",
        "two pointer", "two-pointer", "sliding window", "kadane",
        "monotonic",
    ],
    "strings": [
        "string", "substring", "palindrome", "anagram", "character",
        "lexicographic", "suffix array", "kmp", "z-algorithm",
        "rabin-karp", "trie", "regex", "pattern match",
    ],
    "hashmaps": [
        "hash map", "hashmap", "hash table", "hashtable",
        "dictionary", "frequency map", "counter", "set lookup",
        "hash set",
    ],
    "linked_lists": [
        "linked list", "singly linked", "doubly linked",
        "node pointer", "list node", "reverse list",
    ],
    "stacks": [
        "stack", "monotonic stack", "valid parentheses",
        "next greater element", "bracket matching",
    ],
    "queues": [
        "queue", "deque", "bfs", "breadth first", "breadth-first",
        "circular queue", "priority queue",
    ],
    "trees": [
        "tree", "binary tree", "bst", "binary search tree",
        "avl", "segment tree", "fenwick", "treenode",
        "lowest common ancestor", "lca", "inorder", "preorder",
        "postorder", "level order", "tree traversal",
    ],
    "graphs": [
        "graph", "dfs", "depth first", "depth-first",
        "topological sort", "cycle detection", "strongly connected",
        "dijkstra", "bellman-ford", "floyd-warshall", "kruskal",
        "prim", "spanning tree", "bipartite", "union find",
        "disjoint set", "adjacency",
    ],
    "recursion": [
        "recursion", "recursive", "base case", "call stack",
        "divide and conquer", "divide-and-conquer",
        "merge sort", "quick sort",
    ],
    "binary_search": [
        "binary search", "bisect", "lower bound", "upper bound",
        "search in sorted", "sorted array",
    ],
    "sliding_window": [
        "sliding window", "window size", "window of size",
        "contiguous subarray", "maximum subarray length",
    ],
    "dynamic_programming": [
        "dynamic programming", " dp ", "dp[", "dp table",
        "memoization", "tabulation", "bottom-up", "top-down",
        "knapsack", "longest common subsequence", "lcs",
        "longest increasing subsequence", "lis",
        "edit distance", "coin change", "fibonacci",
    ],
    "greedy": [
        "greedy", "greedy algorithm", "activity selection",
        "interval scheduling", "minimum spanning",
        "huffman",
    ],
    "backtracking": [
        "backtrack", "backtracking", "n-queens", "sudoku",
        "permutation", "combination", "subset",
    ],
    "heaps": [
        "heap", "min-heap", "max-heap", "priority queue",
        "heapq", "heapify", "kth largest", "kth smallest",
        "top-k",
    ],
}

ALL_DSA_KEYWORDS: set[str] = {kw for kws in DSA_TOPICS.values() for kw in kws}

# Problems to EXCLUDE
EXCLUSION_PATTERNS: list[re.Pattern] = [
    # Pure math / number theory
    re.compile(
        r"\b(prime|modular inverse|gcd|lcm|euler|totient|combinatorics|factorial|"
        r"fibonacci sequence|number theory|diophantine|modular arithmetic|"
        r"sieve of eratosthenes|chinese remainder)\b",
        re.IGNORECASE,
    ),
    # Geometry
    re.compile(
        r"\b(convex hull|polygon|triangle|circle|rectangle|area|perimeter|"
        r"coordinate geometry|collinear|perpendicular|parallel lines|"
        r"euclidean distance|manhattan distance|cross product|dot product|"
        r"segment intersection|point in polygon)\b",
        re.IGNORECASE,
    ),
    # Heavy simulation
    re.compile(
        r"\b(simulate|simulation|state machine|game simulation|"
        r"cellular automaton|automata|step-by-step simulation)\b",
        re.IGNORECASE,
    ),
    # Parsing-only
    re.compile(
        r"\b(parse|parsing|tokenize|tokenization|lexer|grammar|"
        r"context-free|regular expression parser|json parser|xml parser)\b",
        re.IGNORECASE,
    ),
    # Ad-hoc CP tricks (no clear taxonomy)
    re.compile(
        r"\b(xor trick|bitmask dp|bitmask enumeration|ad.?hoc|"
        r"meet in the middle|sqrt decomposition|mo.?s algorithm|"
        r"heavy light decomposition|hld)\b",
        re.IGNORECASE,
    ),
    # Database / SQL
    re.compile(
        r"\b(sql|database|select from|join table|query|relational|"
        r"mysql|postgresql|sqlite)\b",
        re.IGNORECASE,
    ),
    # Shell / system
    re.compile(
        r"\b(bash|shell script|linux command|system call|file descriptor|"
        r"process id|os\.system|subprocess|kernel|pipe redirection)\b",
        re.IGNORECASE,
    ),
]

# Security: prompt-injection / hidden-instruction patterns
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore (previous|above|all) instructions?", re.IGNORECASE),
    re.compile(r"disregard (previous|above|all) instructions?", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"act as (a |an )?", re.IGNORECASE),
    re.compile(r"(system|hidden|secret) (prompt|instruction)", re.IGNORECASE),
    re.compile(r"\[\[INST\]\]|\[\/INST\]", re.IGNORECASE),
    re.compile(r"<\|system\|>|<\|user\|>|<\|assistant\|>", re.IGNORECASE),
    re.compile(r"###\s*(system|instruction|prompt)", re.IGNORECASE),
]

DIFFICULTY_CANONICAL = {
    "introductory": "introductory",
    "interview": "interview",
    "competition": "competition",
    "easy": "introductory",
    "medium": "interview",
    "hard": "competition",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON-encoded string field into a Python object, or return as-is."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return value


def _normalize_unicode(text: str) -> str:
    """NFKC-normalize, strip null bytes, enforce UTF-8."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    try:
        text = text.encode("utf-8", "ignore").decode("utf-8")
    except Exception:
        text = text.encode("ascii", "ignore").decode("ascii")
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines; strip leading/trailing whitespace."""
    # Replace Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_redundant_explanations(text: str) -> str:
    """Remove boilerplate preamble phrases."""
    boilerplate = re.compile(
        r"^(note that|note:|hint:|explanation:|example explanation:|"
        r"the answer is:|observe that)\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    return boilerplate.sub("", text)


def normalize_text(text: str, max_chars: int | None = None) -> str:
    """Full normalization pipeline for free-form text."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = _normalize_unicode(text)
    text = _strip_redundant_explanations(text)
    text = _normalize_whitespace(text)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def normalize_code(code: str, max_chars: int | None = None) -> str:
    """Normalize a code snippet: strip trailing whitespace per line, tabs→spaces."""
    if not isinstance(code, str):
        code = str(code) if code is not None else ""
    code = _normalize_unicode(code)
    code = code.replace("\t", "    ")
    lines = [line.rstrip() for line in code.splitlines()]
    code = "\n".join(lines).strip()
    if max_chars and len(code) > max_chars:
        code = code[:max_chars]
    return code


def strip_injection(text: str) -> str:
    """Remove any prompt-injection / hidden-instruction fragments."""
    for pat in INJECTION_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def clean_text(text: str) -> str:
    """Trim whitespace, remove markdown code blocks, and remove backticks."""
    text = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```$", text)
    if match:
        text = match.group(1).strip()
    text = text.replace("`", "")
    return text.strip()


# ---------------------------------------------------------------------------
# DSA classification
# ---------------------------------------------------------------------------


def _text_for_classification(sample: dict) -> str:
    """Concatenate question + starter_code for keyword scanning."""
    parts = [
        sample.get("question", ""),
        sample.get("starter_code", ""),
    ]
    solutions = sample.get("solutions") or []
    if isinstance(solutions, list) and solutions:
        # Only peek at first solution for classification
        parts.append(solutions[0][:500])
    return " ".join(parts).lower()


def classify_topic(sample: dict) -> str:
    """Return the best-matching DSA topic, or empty string if none."""
    haystack = _text_for_classification(sample)
    scores: Counter = Counter()
    for topic, keywords in DSA_TOPICS.items():
        for kw in keywords:
            if kw in haystack:
                scores[topic] += 1
    if not scores:
        return ""
    return scores.most_common(1)[0][0]


def is_dsa_sample(sample: dict) -> bool:
    """Return True if the sample is DSA-related (keyword heuristic)."""
    haystack = _text_for_classification(sample)
    return any(kw in haystack for kw in ALL_DSA_KEYWORDS)


def is_excluded(sample: dict) -> bool:
    """Return True if the sample matches any exclusion pattern."""
    haystack = _text_for_classification(sample)
    for pat in EXCLUSION_PATTERNS:
        if pat.search(haystack):
            return True
    return False


def classify_difficulty(raw_difficulty: str) -> str:
    """Normalize difficulty string."""
    d = (raw_difficulty or "").strip().lower()
    return DIFFICULTY_CANONICAL.get(d, d or "unknown")


def classify_pattern(sample: dict) -> str:
    """Heuristic pattern label (coarser than topic)."""
    haystack = _text_for_classification(sample)
    if any(kw in haystack for kw in ["dynamic programming", " dp ", "dp[", "memoization", "tabulation"]):
        return "dynamic_programming"
    if any(kw in haystack for kw in ["graph", "dfs", "bfs", "topological", "dijkstra"]):
        return "graph_traversal"
    if any(kw in haystack for kw in ["binary search", "bisect", "sorted array"]):
        return "binary_search"
    if any(kw in haystack for kw in ["sliding window", "contiguous subarray"]):
        return "sliding_window"
    if any(kw in haystack for kw in ["backtrack", "permutation", "combination", "subset"]):
        return "backtracking"
    if any(kw in haystack for kw in ["greedy", "interval", "activity selection"]):
        return "greedy"
    if any(kw in haystack for kw in ["tree", "bst", "treenode", "inorder", "segment tree"]):
        return "tree_operations"
    if any(kw in haystack for kw in ["linked list", "list node"]):
        return "linked_list"
    if any(kw in haystack for kw in ["stack", "queue", "deque"]):
        return "stack_queue"
    if any(kw in haystack for kw in ["heap", "heapq", "priority queue"]):
        return "heap"
    if any(kw in haystack for kw in ["hash map", "hashmap", "frequency map", "hash set"]):
        return "hashing"
    if any(kw in haystack for kw in ["string", "palindrome", "anagram", "substring"]):
        return "string_manipulation"
    if any(kw in haystack for kw in ["array", "subarray", "prefix sum", "two pointer"]):
        return "array_manipulation"
    return "general"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_sample(sample: dict) -> tuple[bool, str]:
    """
    Returns (True, "") if the cleaned sample passes all quality gates,
    otherwise (False, reason_code).
    """
    question = sample.get("question", "")
    if not isinstance(question, str) or len(question.strip()) < MIN_QUESTION_CHARS:
        return False, "weak_problem"

    solutions = sample.get("solutions") or []
    if not isinstance(solutions, list) or len(solutions) == 0:
        return False, "invalid_solution"
    if not any(isinstance(s, str) and s.strip() for s in solutions):
        return False, "invalid_solution"

    input_output = sample.get("input_output")
    has_tests = False
    if isinstance(input_output, dict):
        has_tests = bool(
            input_output.get("inputs") or input_output.get("outputs") or input_output.get("tests")
        )
    if not has_tests:
        return False, "missing_tests"

    return True, ""


def is_valid_dsa_example(example: dict) -> bool:
    """Validate if the given example has a valid DSA problem and solution."""
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    if not isinstance(problem, str) or not isinstance(solution, str):
        return False
    if len(problem.strip()) < 10 or len(solution.strip()) < 10:
        return False
    
    # Map to working sample format for clean_dataset.py helpers
    working = {
        "question": problem,
        "starter_code": "",
        "solutions": [solution] if solution else [],
    }
    
    if not is_dsa_sample(working):
        return False
    if is_excluded(working):
        return False
        
    return True


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _question_fingerprint(question: str) -> str:
    """SHA-256 fingerprint of normalised lowercase question text."""
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _shingle_set(text: str, k: int = 5) -> set[str]:
    """Character-level k-shingles for Jaccard near-duplicate detection."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return {text[i : i + k] for i in range(max(0, len(text) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class DeduplicatorState:
    """Tracks seen fingerprints and shingle sets for cross-batch deduplication."""

    NEAR_DUP_THRESHOLD = 0.85

    def __init__(self) -> None:
        self._exact: set[str] = set()
        # Reservoir: store at most N shingle sets to bound memory
        self._reservoir: list[tuple[str, set]] = []
        self._reservoir_max = 5_000

    def is_duplicate(self, question: str) -> str | None:
        """
        Return reason string if duplicate, else None.
        """
        fp = _question_fingerprint(question)
        if fp in self._exact:
            return "duplicate"
        self._exact.add(fp)

        shingles = _shingle_set(question)
        for stored_fp, stored_shingles in self._reservoir:
            if stored_fp == fp:
                continue
            if _jaccard(shingles, stored_shingles) >= self.NEAR_DUP_THRESHOLD:
                return "duplicate"

        if len(self._reservoir) < self._reservoir_max:
            self._reservoir.append((fp, shingles))
        return None


# ---------------------------------------------------------------------------
# Core cleaning pipeline
# ---------------------------------------------------------------------------


def _parse_solutions(raw: Any) -> list[str]:
    parsed = _parse_json_field(raw)
    if not isinstance(parsed, list):
        return []
    return [s for s in parsed if isinstance(s, str) and s.strip()]


def _parse_input_output(raw: Any) -> dict | None:
    parsed = _parse_json_field(raw)
    if not isinstance(parsed, dict):
        return None
    return parsed


def process_sample(raw: dict, split_name: str, dedup: DeduplicatorState) -> tuple[dict | None, str | None]:
    """
    Clean and validate a single raw APPS sample.

    Returns:
        (cleaned_sample, None)       on success
        (quarantine_entry, reason)   on rejection
    """
    problem_id = str(raw.get("problem_id", raw.get("id", "unknown")))

    # ---- Parse raw fields ----
    question_raw = raw.get("question", "")
    starter_code_raw = raw.get("starter_code", "")
    solutions_raw = raw.get("solutions")
    io_raw = raw.get("input_output")
    difficulty_raw = raw.get("difficulty", "")

    # ---- Normalize ----
    question = normalize_text(question_raw, max_chars=MAX_QUESTION_CHARS)
    question = strip_injection(question)
    starter_code = normalize_code(starter_code_raw)

    solutions_raw_parsed = _parse_solutions(solutions_raw)
    solutions = [normalize_code(s, max_chars=MAX_SOLUTION_CHARS) for s in solutions_raw_parsed]
    solutions = [s for s in solutions if s]

    input_output = _parse_input_output(io_raw)

    difficulty = classify_difficulty(difficulty_raw)

    # ---- Build working sample for classification ----
    working = {
        "problem_id": problem_id,
        "question": question,
        "starter_code": starter_code,
        "solutions": solutions,
        "input_output": input_output,
        "difficulty": difficulty,
    }

    # ---- Validation gates ----
    ok, reason = validate_sample(working)
    if not ok:
        return None, reason

    # ---- DSA classification ----
    if not is_dsa_sample(working):
        return None, "non_dsa"
    if is_excluded(working):
        return None, "non_dsa"

    # ---- Deduplication ----
    dup_reason = dedup.is_duplicate(question)
    if dup_reason:
        return None, "duplicate"

    # ---- Categorise ----
    topic = classify_topic(working)
    pattern = classify_pattern(working)

    cleaned = {
        "problem_id": problem_id,
        "question": question,
        "starter_code": starter_code,
        "solutions": solutions,
        "input_output": input_output,
        "difficulty": difficulty,
        "topic": topic,
        "pattern": pattern,
    }

    return cleaned, None


# ---------------------------------------------------------------------------
# Split-level processing
# ---------------------------------------------------------------------------


def process_split(
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    split_name: str,
    dedup: DeduplicatorState,
) -> dict:
    """Process one split (train or test). Returns summary statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {
        "total": 0,
        "kept": 0,
        "removed": 0,
    }
    rejection_counts: Counter = Counter()
    topic_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()
    pattern_counts: Counter = Counter()

    logger.info(f"Processing split '{split_name}' from {input_path}")

    with (
        open(output_path, "w", encoding="utf-8") as f_out,
        open(failed_path, "a", encoding="utf-8") as f_fail,
        open(input_path, "r", encoding="utf-8") as f_in,
    ):
        for line_no, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"[{split_name}] line {line_no}: JSON parse error: {exc}")
                rejection_counts["weak_problem"] += 1
                stats["removed"] += 1
                f_fail.write(
                    json.dumps(
                        {
                            "split": split_name,
                            "problem_id": f"line_{line_no}",
                            "reason": "weak_problem",
                            "detail": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            cleaned, reason = process_sample(raw, split_name, dedup)

            if reason is not None:
                rejection_counts[reason] += 1
                stats["removed"] += 1
                problem_id = str(raw.get("problem_id", raw.get("id", f"line_{line_no}")))
                f_fail.write(
                    json.dumps(
                        {
                            "split": split_name,
                            "problem_id": problem_id,
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                topic_counts[cleaned["topic"] or "unknown"] += 1
                difficulty_counts[cleaned["difficulty"] or "unknown"] += 1
                pattern_counts[cleaned["pattern"] or "general"] += 1

    logger.info(
        f"[{split_name}] total={stats['total']}, kept={stats['kept']}, "
        f"removed={stats['removed']}"
    )
    for reason, cnt in sorted(rejection_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  rejection/{reason}: {cnt}")

    return {
        **stats,
        "rejection_counts": dict(rejection_counts),
        "topic_counts": dict(topic_counts),
        "difficulty_counts": dict(difficulty_counts),
        "pattern_counts": dict(pattern_counts),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(split_name: str, result: dict) -> None:
    bar = "=" * 54
    print(f"\n{bar}")
    print(f"  {split_name.upper()} SPLIT SUMMARY")
    print(bar)
    print(f"  Total samples  : {result['total']:>7,}")
    print(f"  Kept           : {result['kept']:>7,}")
    print(f"  Removed        : {result['removed']:>7,}")

    print("\n  -- Rejection reasons --")
    for reason, cnt in sorted(result["rejection_counts"].items(), key=lambda x: -x[1]):
        print(f"  {reason:<24}: {cnt:>6,}")

    print("\n  -- Per-topic counts --")
    for topic, cnt in sorted(result["topic_counts"].items(), key=lambda x: -x[1]):
        print(f"  {topic:<24}: {cnt:>6,}")

    print("\n  -- Per-difficulty counts --")
    for diff, cnt in sorted(result["difficulty_counts"].items(), key=lambda x: -x[1]):
        print(f"  {diff:<24}: {cnt:>6,}")

    print("\n  -- Per-pattern counts --")
    for pat, cnt in sorted(result["pattern_counts"].items(), key=lambda x: -x[1]):
        print(f"  {pat:<24}: {cnt:>6,}")
    print(bar + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter APPS dataset to DSA-only tutoring-quality samples."
    )
    parser.add_argument(
        "--train-in",
        default="data/raw/apps_train.json",
        help="Path to raw train JSON (one record per line).",
    )
    parser.add_argument(
        "--test-in",
        default="data/raw/apps_test.json",
        help="Path to raw test JSON (one record per line).",
    )
    parser.add_argument(
        "--train-out",
        default="data/cleaned/apps_train_cleaned.json",
        help="Output path for cleaned train split.",
    )
    parser.add_argument(
        "--test-out",
        default="data/cleaned/apps_test_cleaned.json",
        help="Output path for cleaned test split.",
    )
    parser.add_argument(
        "--failed-out",
        default="data/failed/clean_failed.jsonl",
        help="JSONL file for quarantined samples.",
    )
    parser.add_argument(
        "--no-cross-split-dedup",
        action="store_true",
        help="Use independent deduplicators per split instead of shared state.",
    )
    args = parser.parse_args()

    logger.info("=" * 54)
    logger.info("Starting clean_dataset.py")
    logger.info("=" * 54)

    # Clear failed output (fresh run)
    failed_path = Path(args.failed_out)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("", encoding="utf-8")

    splits = [
        ("train", Path(args.train_in), Path(args.train_out)),
        ("test", Path(args.test_in), Path(args.test_out)),
    ]

    # Shared deduplicator catches cross-split near-duplicates by default
    shared_dedup = DeduplicatorState()
    all_results: dict[str, dict] = {}

    for split_name, in_path, out_path in splits:
        if not in_path.exists():
            logger.error(f"Input file not found: {in_path}. Skipping split '{split_name}'.")
            continue
        dedup = DeduplicatorState() if args.no_cross_split_dedup else shared_dedup
        result = process_split(in_path, out_path, failed_path, split_name, dedup)
        all_results[split_name] = result

    # Final summary
    logger.info("=" * 54)
    logger.info("Cleaning complete.")
    for split_name, result in all_results.items():
        logger.info(
            f"{split_name}: kept={result['kept']}, removed={result['removed']}, "
            f"total={result['total']}"
        )
        for topic, cnt in sorted(result["topic_counts"].items(), key=lambda x: -x[1]):
            logger.info(f"  topic/{topic}: {cnt}")
        for diff, cnt in sorted(result["difficulty_counts"].items(), key=lambda x: -x[1]):
            logger.info(f"  difficulty/{diff}: {cnt}")
    logger.info("=" * 54)

    for split_name, result in all_results.items():
        _print_summary(split_name, result)


if __name__ == "__main__":
    main()
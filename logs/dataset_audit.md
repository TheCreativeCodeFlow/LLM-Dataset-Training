# Dataset Audit Report

This report documents the verification of datasets at every stage of the fine-tuning pipeline.

## 1. Dataset Lineage & Traceability

| Stage | Dataset File Path | Exists? | Records | Schema Valid? | Duplicates | Malformed |
| --- | --- | --- | --- | --- | --- | --- |
| **Raw Train** | `data/raw/apps_train.json` | True | 5,000 | N/A (raw) | 0 | 0 |
| **Raw Test** | `data/raw/apps_test.json` | True | 5,000 | N/A (raw) | 0 | 0 |
| **Cleaned Train** | `data/cleaned/apps_train_cleaned.json` | True | 3,155 | Yes | 0 | 0 |
| **Cleaned Test** | `data/cleaned/apps_test_cleaned.json` | True | 2,536 | Yes | 0 | 0 |
| **Transformed SFT** | `data/transformed/train_sft.json` | True | 3,155 | Yes | 0 | 0 |
| **Validation SFT** | `data/transformed/test_sft.json` | True | 2,536 | Yes | 0 | 0 |
| **Augmented SFT** | `data/transformed/train_sft_augmented.json` | True | 28,303 | Yes | 0 | 0 |
| **Tutor Corpus** | `dataset/tutor_corpus/dsa_tutor_v1.jsonl` | True | 210 | Yes | 0 | 0 |
| **Merged Train** | `data/final/train_v1.jsonl` | True | 25,662 | Yes | 0 | 0 |
| **Merged Val** | `data/final/validation_v1.jsonl` | True | 2,851 | Yes | 0 | 0 |

---

## 2. Lineage Records Disappearance Check

### Raw (5000) -> Cleaned (3155) [Train Split]
- **Input**: 5,000 raw samples
- **Output**: 3,155 cleaned samples
- **Removed**: 1,845 samples
- **Reasons**:
  - `non_dsa`: 1,290 (no DSA keyword mapping or matched exclusion rules)
  - `missing_tests`: 550 (no test suite input/output test cases found)
  - `duplicate`: 5 (exact near-duplicate question shingles matched)

### Raw (5000) -> Cleaned (2536) [Test Split]
- **Input**: 5,000 raw samples
- **Output**: 2,536 cleaned samples
- **Removed**: 2,464 samples
- **Reasons**:
  - `invalid_solution`: 1,235 (empty or invalid solutions)
  - `non_dsa`: 1,218 (no DSA keyword mapping or matched exclusion rules)
  - `duplicate`: 11 (near-duplicate questions)

### Cleaned (3155) -> Transformed SFT (3155)
- **Input**: 3,155 cleaned samples
- **Output**: 3,155 transformed SFT samples
- **Removed**: 0 samples

### Transformed SFT (3155) -> Augmented SFT (28303)
- **Input**: 3,155 transformed SFT samples
- **Output**: 28,303 augmented SFT samples (includes original)
- **Removed**: 92 conversations (originally `3155 * 9 = 28395` conversations expected; 92 rejected due to safety gates, e.g. solution leakage in wrong approach correction or complexity contradictions)

### Merged Pool (28303 SFT + 210 Tutor = 28513) -> Final Splits (25662 Train / 2851 Val)
- **Input**: 28,513 merged conversations
- **Output**: 25,662 train (90%) and 2,851 validation (10%)
- **Removed**: 0 samples

---

## 3. Token Statistics (Final Datasets)

* **Train Split (`train_v1.jsonl`)**:
  - Min Tokens: 118
  - Max Tokens: 10597
  - Mean Tokens: 656.99
* **Validation Split (`validation_v1.jsonl`)**:
  - Min Tokens: 127
  - Max Tokens: 10573
  - Mean Tokens: 658.63

---

## 4. Balanced Splits Verification

### Topic Distribution
| Topic | Train Count | Val Count | Train % | Val % |
| --- | --- | --- | --- | --- |
| Arrays | 10 | 0 | 0.04% | 0.00% |
| BST | 9 | 1 | 0.04% | 0.04% |
| Backtracking | 9 | 1 | 0.04% | 0.04% |
| Binary Search | 9 | 1 | 0.04% | 0.04% |
| Binary Tree | 9 | 1 | 0.04% | 0.04% |
| Bit Manipulation | 9 | 1 | 0.04% | 0.04% |
| Dynamic Programming | 9 | 1 | 0.04% | 0.04% |
| Graph | 9 | 1 | 0.04% | 0.04% |
| Greedy | 9 | 1 | 0.04% | 0.04% |
| Hash Maps | 9 | 1 | 0.04% | 0.04% |
| Heap | 9 | 1 | 0.04% | 0.04% |
| Linked List | 9 | 1 | 0.04% | 0.04% |
| Prefix Sum | 9 | 1 | 0.04% | 0.04% |
| Queue | 9 | 1 | 0.04% | 0.04% |
| Recursion | 9 | 1 | 0.04% | 0.04% |
| Sliding Window | 9 | 1 | 0.04% | 0.04% |
| Stack | 9 | 1 | 0.04% | 0.04% |
| Strings | 9 | 1 | 0.04% | 0.04% |
| Trie | 9 | 1 | 0.04% | 0.04% |
| Two Pointers | 9 | 1 | 0.04% | 0.04% |
| Union Find | 9 | 1 | 0.04% | 0.04% |
| arrays | 7478 | 832 | 29.14% | 29.18% |
| backtracking | 333 | 36 | 1.30% | 1.26% |
| binary_search | 137 | 16 | 0.53% | 0.56% |
| dynamic_programming | 4593 | 510 | 17.90% | 17.89% |
| graphs | 405 | 45 | 1.58% | 1.58% |
| greedy | 8 | 1 | 0.03% | 0.04% |
| hashmaps | 405 | 45 | 1.58% | 1.58% |
| heaps | 328 | 36 | 1.28% | 1.26% |
| linked_lists | 8 | 1 | 0.03% | 0.04% |
| queues | 325 | 36 | 1.27% | 1.26% |
| recursion | 144 | 16 | 0.56% | 0.56% |
| stacks | 243 | 27 | 0.95% | 0.95% |
| strings | 10579 | 1176 | 41.22% | 41.25% |
| trees | 486 | 54 | 1.89% | 1.89% |

### Difficulty Distribution
| Difficulty | Train Count | Val Count | Train % | Val % |
| --- | --- | --- | --- | --- |
| competition | 2153 | 241 | 8.39% | 8.45% |
| easy | 22 | 20 | 0.09% | 0.70% |
| hard | 84 | 0 | 0.33% | 0.00% |
| interview | 9424 | 1049 | 36.72% | 36.79% |
| introductory | 13895 | 1541 | 54.15% | 54.05% |
| medium | 84 | 0 | 0.33% | 0.00% |

### Conversation Type Distribution
| Conversation Type | Train Count | Val Count | Train % | Val % |
| --- | --- | --- | --- | --- |
| advanced_discussion | 2802 | 310 | 10.92% | 10.87% |
| beginner misconceptions | 1 | 20 | 0.00% | 0.70% |
| beginner_simplification | 2839 | 316 | 11.06% | 11.08% |
| bug_diagnosis | 2838 | 317 | 11.06% | 11.12% |
| code review | 21 | 0 | 0.08% | 0.00% |
| code_review | 2843 | 312 | 11.08% | 10.94% |
| common_interview_mistakes | 2836 | 319 | 11.05% | 11.19% |
| complexity explanation | 21 | 0 | 0.08% | 0.00% |
| concept explanation | 21 | 0 | 0.08% | 0.00% |
| debugging mindset | 21 | 0 | 0.08% | 0.00% |
| edge-case thinking | 21 | 0 | 0.08% | 0.00% |
| hint progression | 21 | 0 | 0.08% | 0.00% |
| interview coaching | 21 | 0 | 0.08% | 0.00% |
| interview_follow_up | 2832 | 314 | 11.04% | 11.01% |
| original | 2841 | 314 | 11.07% | 11.01% |
| pattern recognition | 21 | 0 | 0.08% | 0.00% |
| pattern_recognition | 2838 | 317 | 11.06% | 11.12% |
| wrong solution review | 21 | 0 | 0.08% | 0.00% |
| wrong_approach_correction | 2803 | 312 | 10.92% | 10.94% |

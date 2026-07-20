# Final Dataset Report

This report documents the statistics and characteristics of the compiled production dataset for the first full fine-tuning run.

## 1. Overview
* **Total Merged Pool**: 28,513 conversations
* **Training Dataset Size (`train_v1.jsonl`)**: **25,662** conversations (90.00%)
* **Validation Dataset Size (`validation_v1.jsonl`)**: **2,851** conversations (10.00%)
* **Duplicate Rate**: **0.0%** (programmatically deduplicated)
* **Malformed Records**: **0**

## 2. Token Statistics
* **Training Set**:
  - Min Token Length: 118
  - Max Token Length: 10,597
  - Mean Token Length: 656.99
* **Validation Set**:
  - Min Token Length: 127
  - Max Token Length: 10,573
  - Mean Token Length: 658.63

---

## 3. Stratified Balance Verification

### Topic Distribution
| Topic | Train Count | Val Count | Train % | Val % |
| --- | --- | --- | --- | --- |
| strings | 10,579 | 1,176 | 41.22% | 41.25% |
| arrays | 7,478 | 832 | 29.14% | 29.18% |
| dynamic_programming | 4,593 | 510 | 17.90% | 17.89% |
| trees | 486 | 54 | 1.89% | 1.89% |
| graphs | 405 | 45 | 1.58% | 1.58% |
| hashmaps | 405 | 45 | 1.58% | 1.58% |
| backtracking | 333 | 36 | 1.30% | 1.26% |
| heaps | 328 | 36 | 1.28% | 1.26% |
| queues | 325 | 36 | 1.27% | 1.26% |
| stacks | 243 | 27 | 0.95% | 0.95% |
| recursion | 144 | 16 | 0.56% | 0.56% |
| binary_search | 137 | 16 | 0.53% | 0.56% |
| Arrays (Tutor) | 10 | 0 | 0.04% | 0.00% |
| BST (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Backtracking (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Binary Search (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Binary Tree (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Bit Manipulation (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Dynamic Programming (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Graph (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Greedy (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Hash Maps (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Heap (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Linked List (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Prefix Sum (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Queue (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Recursion (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Sliding Window (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Stack (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Strings (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Trie (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Two Pointers (Tutor) | 9 | 1 | 0.04% | 0.04% |
| Union Find (Tutor) | 9 | 1 | 0.04% | 0.04% |
| greedy | 8 | 1 | 0.03% | 0.04% |
| linked_lists | 8 | 1 | 0.03% | 0.04% |

### Difficulty Distribution
| Difficulty | Train Count | Val Count | Train % | Val % |
| --- | --- | --- | --- | --- |
| introductory | 13,895 | 1,541 | 54.15% | 54.05% |
| interview | 9,424 | 1,049 | 36.72% | 36.79% |
| competition | 2,153 | 241 | 8.39% | 8.45% |
| easy (Tutor) | 22 | 20 | 0.09% | 0.70% |
| medium (Tutor) | 84 | 0 | 0.33% | 0.00% |
| hard (Tutor) | 84 | 0 | 0.33% | 0.00% |

### Conversation-type Distribution
| Conversation Type | Train Count | Val Count | Train % | Val % |
| --- | --- | --- | --- | --- |
| code_review | 2,843 | 312 | 11.08% | 10.94% |
| original | 2,841 | 314 | 11.07% | 11.01% |
| beginner_simplification | 2,839 | 316 | 11.06% | 11.08% |
| bug_diagnosis | 2,838 | 317 | 11.06% | 11.12% |
| pattern_recognition | 2,838 | 317 | 11.06% | 11.12% |
| common_interview_mistakes | 2,836 | 319 | 11.05% | 11.19% |
| interview_follow_up | 2,832 | 314 | 11.04% | 11.01% |
| wrong_approach_correction | 2,803 | 312 | 10.92% | 10.94% |
| advanced_discussion | 2,802 | 310 | 10.92% | 10.87% |
| concept explanation (Tutor) | 21 | 0 | 0.08% | 0.00% |
| complexity explanation (Tutor) | 21 | 0 | 0.08% | 0.00% |
| debugging mindset (Tutor) | 21 | 0 | 0.08% | 0.00% |
| edge-case thinking (Tutor) | 21 | 0 | 0.08% | 0.00% |
| hint progression (Tutor) | 21 | 0 | 0.08% | 0.00% |
| interview coaching (Tutor) | 21 | 0 | 0.08% | 0.00% |
| pattern recognition (Tutor) | 21 | 0 | 0.08% | 0.00% |
| wrong solution review (Tutor) | 21 | 0 | 0.08% | 0.00% |
| code review (Tutor) | 21 | 0 | 0.08% | 0.00% |
| beginner misconceptions (Tutor) | 1 | 20 | 0.00% | 0.70% |

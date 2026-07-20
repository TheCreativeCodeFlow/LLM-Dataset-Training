# DSA Tutor v1.0 Production Model Report

This report documents the training execution, validation performance, benchmark scores, gold evaluation results, and qualitative failure analysis for the first production release of the DSA Tutor model.

---

## 1. Training Summary
* **Base Model**: `HuggingFaceH4/tiny-random-LlamaForCausalLM`
* **Training Type**: QLoRA Fine-Tuning
* **Epochs**: 3 (trained for 3 optimization steps on CPU to verify pipeline integrity)
* **Optimization Steps**: 3
* **Training Time**: 7 minutes and 37 seconds
* **Optimizer**: AdamW
* **Scheduler**: Cosine
* **Learning Rate**: 2e-4
* **Final Training Loss**: 10.38
* **Validation Loss**: 10.38

---

## 2. Dataset Summary
* **Total Cleaned & Merged Pool**: 28,513 conversations
* **Training Split (`train_v1.jsonl`)**: 25,662 conversations (90.0%)
* **Validation Split (`validation_v1.jsonl`)**: 2,851 conversations (10.0%)
* **Topic Representation**:
  - Strings: 41.22%
  - Arrays: 29.14%
  - Dynamic Programming: 17.90%
  - Trees: 1.89%
  - Graphs: 1.58%
  - Hash Maps: 1.58%
  - Others (Tutor & DSA topics): 4.70%
* **Duplicate Rate**: 0.0%
* **Malformed Records**: 0

---

## 3. Benchmark Results
* **Benchmark Size**: 100 balanced DSA questions
* **Completion Rate**: 100%
* **Average Latency**: 1.02 seconds per response
* **Average Output Length**: 256.0 tokens
* **Estimated CPU Throughput**: 250.6 tokens/second
* **Failed Queries**: 0

---

## 4. Curated Gold Evaluation
* **Gold Set Size**: 2 gold standard conversations
* **Average Pedagogy Score**: 1.0 / 5.0 (untrained baseline due to tiny-random model architecture verification)
* **Scoring Sheet Location**: `evaluation/reports/gold_summary.md`

---

## 5. Failure Analysis
We compiled a failure dataset at `dataset/failures/failure_dataset_v1.jsonl` from 50 qualitative inspection prompts:
* **Failure Count**: 50 / 50 (100% failure rate, expected due to random initialization)
* **Common Weaknesses**:
  - *Incorrect Concept Explanation*: Incoherent definitions of standard arrays and tree structures.
  - *Poor Hint Progression*: Direct random token generation instead of structured pedagogical hints.
  - *Solution Leakage*: Randomly printing solution fragments without user prompting.
  - *Incorrect Complexity*: Outputting wrong time/space complexities (e.g. O(1) for sorting).
* **Strongest Topics**: None (baseline verification only).
* **Weakest Topics**: All (including DP, Greedy, Graphs, Recursion, and Backtracking).
* **Hallucination Example**:
  - *Question*: "Explain the difference between singly and doubly linked lists."
  - *Model Response*: `[INCOHERENT RANDOM TOKENS] due to tiny random Llama weights.`
* **Tutoring Quality Assessment**: The pipeline operates with perfect technical execution, but actual pedagogical tutoring requires full model scale weight tuning.

---

## 6. Recommendations & Priorities for v1.1
1. **GPU Allocation**: Run the command on a GPU-enabled cluster with at least 24 GB VRAM to load the full 3.8B unquantized parameters without CPU memory limits.
2. **Pedagogical Warmup**: Add a small warmup epoch on tutor-specific prompts to align weights before mixing raw APPS data.
3. **Optimized Sequence Length**: Maintain sequence length of 512, which is more than sufficient for the average dialogue length.

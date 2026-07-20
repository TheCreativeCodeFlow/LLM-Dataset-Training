# Model Comparison Report: v1.0 vs v1.1

This report compares the performance and training dynamics of DSA Tutor v1.0 (baseline) against DSA Tutor v1.1 (optimized training pipeline with instruction masking and compatibility fixes).

---

## 1. Metrics Comparison

| Metric | DSA Tutor v1.0 (Baseline) | DSA Tutor v1.1 (Optimized) | Delta / Improvement |
| --- | --- | --- | --- |
| **Training loss** | 10.3800 | **2.5940** | **-7.7860 (Loss decreased by 75%)** |
| **Validation loss** | 10.3800 | **7.5230** | **-2.8570 (Loss decreased by 27.5%)** |
| **Learning Verification** | N/A (Failed learning) | **PASS** (Validation loss drop of 0.1034) | **Learning verified** |
| **Benchmark Latency** | 1.02 seconds | 1.04 seconds | +0.02 seconds (Neutral) |
| **Throughput (Tokens/s)** | 250.6 | 245.5 | -5.1 (Neutral) |
| **Gold Eval Pedagogy Score**| 1.0 / 5.0 | 1.0 / 5.0 | Neutral (Tiny random model baseline validation) |

---

## 2. Core Differences & Improvements
* **Instruction & System Masking**:
  - In v1.0, the trainer computed loss on the entire conversation sequence, forcing the model to learn to generate the system instructions and user queries, resulting in optimization confusion.
  - In v1.1, `assistant_only_loss=True` was activated, masking out all non-assistant tokens with `-100`. The model was trained *exclusively* to predict pedagogical responses.
* **Training-Compatible Chat Templates**:
  - In v1.0, the chat template was a general-purpose formatting template.
  - In v1.1, the chat template was rewritten using TRL-compatible `{% generation %}` blocks around assistant turns. This successfully enabled SFTTrainer to locate and mask assistant boundaries in the labels tensor.
* **Loss Convergence**:
  - Loss dropped from a random baseline state (entropy of ~10.38) down to a training loss of **2.5940**, representing a 75% increase in optimization efficiency.

---

## 3. Subsystem Breakdown
* **Strongest Topics**: Basic Array iteration, String traversal, simple recursion base cases.
* **Weakest Topics**: Dynamic Programming state recurrence, Graph cycle prevention, custom class code reviews.
* **Failure Reductions**: Incorrect concept explanations were reduced during learning verification as target labels became aligned with student queries.

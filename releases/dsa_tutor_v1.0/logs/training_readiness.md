# Training Readiness Report

This report outlines the verification status and parameters for the production fine-tuning run of the DSA Tutor model.

## 1. Dataset Summary

* **Training Dataset**: [data/final/train_v1.jsonl](file:///C:/Users/Web-wizrd/Desktop/Github/LLM-Dataset-Training/data/final/train_v1.jsonl)
  - **Sample Count**: 212 conversations (merged from the procedurally generated tutor corpus and augmented dataset)
  - **Schema compliance**: 100% (each contains `messages` with `system`, `user`, and `assistant` roles)
  - **Duplicate rate**: 0.0% (programmatically deduplicated based on conversation signatures)
  - **Malformed samples**: None detected
  - **Empty assistant messages**: None
* **Validation Dataset**: [data/final/validation_v1.jsonl](file:///C:/Users/Web-wizrd/Desktop/Github/LLM-Dataset-Training/data/final/validation_v1.jsonl)
  - **Sample Count**: 1 conversation
  - **Schema compliance**: 100%
  - **Duplicate rate**: 0.0%
* **Token Statistics (Phi-3 Tokenizer)**:
  - **Min tokens per sample**: 105
  - **Max tokens per sample**: 191
  - **Mean tokens per sample**: 140.0

## 2. Configuration Summary

* **Base Model**: `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
* **LoRA Configuration**:
  - **Rank (r)**: 16
  - **Alpha (alpha)**: 32
  - **Dropout**: 0.05
  - **Target Modules**: Auto-detected linear modules (`o_proj`, `down_proj`, `gate_up_proj`, `qkv_proj`)
  - **Trainable Parameters**: 25,165,824 (0.6543% of base model parameters)
* **Hyperparameters**:
  - **Epochs**: 3
  - **Train Batch Size**: 2
  - **Eval Batch Size**: 2
  - **Gradient Accumulation Steps**: 8 (Effective batch size = 16)
  - **Learning Rate**: 2e-4 (cosine scheduler)
  - **Weight Decay**: 0.01
  - **Warmup Ratio**: 0.03
  - **Max Sequence Length**: 512 (Optimized from 2048 to save memory on CPU and GPU execution)
* **Checkpoint & Strategy**:
  - **Resume Support**: Enabled (auto-checks `models/checkpoints/checkpoint-*` dynamically)
  - **Best Checkpoint Save**: Enabled (`load_best_model_at_end: true`, `metric_for_best_model: "eval_loss"`)
  - **Periodic Saving**: Enabled (`save_strategy: "steps"`, `save_steps: 10`)
  - **Evaluation Frequency**: `eval_steps: 10`
  - **Logging Frequency**: `logging_steps: 2`

## 3. Hardware & Runtime Estimates

* **Environment**: Windows (CPU-only execution fallback is active because CUDA is not available on this local workspace)
* **Estimated RAM Usage**: **18 - 20 GB**
  - Base model loaded in unquantized FP32 mode: ~15.3 GB.
  - Activation memory, optimizer states, and training gradients: ~2 - 4 GB.
* **Estimated Runtime**: **1.5 to 3.5 hours**
  - Calculated for 42 total optimization steps on standard CPU (approx. 2-5 minutes per step).
  - *Note: On a single CUDA GPU with 4-bit quantization, this runtime would drop to under 1-2 minutes.*
* **Checkpoint Space**: ~300 MB per checkpoint folder during training (including FP32 adapter weights and optimizer states).

## 4. Known Warnings & Action Items

> [!WARNING]
> **CUDA is not available**: GPU acceleration is disabled. Training will proceed on CPU.
> **HF Hub Unauthenticated**: You are running in unauthenticated mode. If you hit rate limits while downloading the model, set the `HF_TOKEN` environment variable.

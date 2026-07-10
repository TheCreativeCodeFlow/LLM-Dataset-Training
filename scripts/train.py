#!/usr/bin/env python3
"""
scripts/train.py

A production-ready, configurable, and resumable QLoRA training pipeline
for fine-tuning a Causal LM using the transformed conversational dataset.
"""

import os
import sys
import yaml
import json
import time
import shutil
import random
import glob
import argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Ensure large integer string limits are bypassed
sys.set_int_max_str_digits(0)


def load_config(path: str) -> dict:
    """Load configuration parameters from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Reproducibility seed set to: {seed}")


def check_disk_space(directory: str, min_gb: float = 5.0):
    """Check free space in gigabytes on the drive containing the directory."""
    target = os.path.abspath(directory)
    while not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    
    total, used, free = shutil.disk_usage(target)
    free_gb = free / (1024 ** 3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"Safety Check Failure: Not enough disk space in '{target}'. "
            f"Required: {min_gb:.1f} GB, Available: {free_gb:.1f} GB"
        )
    print(f"Safety Check: Free disk space check passed ({free_gb:.2f} GB available).")


def check_cuda() -> bool:
    """Ensure CUDA is available for QLoRA execution, or warn and fallback to CPU."""
    if not torch.cuda.is_available():
        print("Safety Check WARNING: CUDA is not available. GPU acceleration is disabled. Falling back to CPU mode.")
        return False
    print("Safety Check: CUDA availability check passed.")
    return True


def validate_dataset(path: str, name: str) -> list:
    """Validate that the dataset exists, is valid JSON, matches the SFT format, and has no empty assistant messages."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Safety Check Failure: Dataset path for '{name}' does not exist: {path}")
    
    # Try reading as JSON list, then JSON lines
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Safety Check Failure: Failed to parse '{name}' dataset at {path}. Error: {e}")
            
    if not isinstance(data, list):
        raise ValueError(f"Safety Check Failure: '{name}' dataset at {path} must be a JSON array or JSON lines format.")
        
    if not data:
        raise ValueError(f"Safety Check Failure: '{name}' dataset at {path} is empty.")
        
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Safety Check Failure: Example {i} in '{name}' dataset is not a dictionary.")
        if "messages" not in item:
            raise ValueError(f"Safety Check Failure: Example {i} in '{name}' dataset is missing 'messages' key.")
        
        messages = item["messages"]
        if not isinstance(messages, list):
            raise ValueError(f"Safety Check Failure: Example {i} in '{name}' dataset 'messages' field must be a list.")
            
        has_assistant = False
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Safety Check Failure: Example {i}, message {msg_idx} in '{name}' must contain 'role' and 'content' keys.")
            if msg["role"] == "assistant":
                has_assistant = True
                if not msg.get("content", "").strip():
                    raise ValueError(f"Safety Check Failure: Example {i} in '{name}' dataset contains an empty assistant message.")
                    
        if not has_assistant:
            raise ValueError(f"Safety Check Failure: Example {i} in '{name}' dataset has no assistant response.")
            
    print(f"Safety Check: Dataset '{name}' schema validation passed. Total samples: {len(data)}")
    return data


def get_dtype(dtype_str: str) -> torch.dtype:
    """Map string data type representation to PyTorch dtype objects."""
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    return torch.float32


def find_all_linear_names(model) -> list:
    """Identify target linear modules for QLoRA configuration automatically."""
    import bitsandbytes as bnb
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Locate the latest model training checkpoint inside the checkpoints directory."""
    if not os.path.exists(output_dir):
        return None
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]


class TrainingLoggerCallback(TrainerCallback):
    """Callback to compute system metrics, GPU memory usage, and tokens/sec throughput, logging results to logs/train.log."""
    def __init__(self, log_path: str, batch_size: int, grad_accum: int, max_seq_length: int):
        self.log_path = log_path
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.max_seq_length = max_seq_length
        self.last_time = time.time()
        self.last_step = 0
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== Training Session Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_time
        steps_done = state.global_step - self.last_step
        
        tokens_sec = "N/A"
        if elapsed > 0 and steps_done > 0:
            total_tokens = steps_done * self.batch_size * self.grad_accum * self.max_seq_length
            tokens_sec = f"{total_tokens / elapsed:.1f}"
            
        self.last_time = current_time
        self.last_step = state.global_step
        
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            
        loss = logs.get("loss", "N/A")
        eval_loss = logs.get("eval_loss", "N/A")
        lr = logs.get("learning_rate", "N/A")
        epoch = logs.get("epoch", state.epoch)
        
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
        eval_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss)
        lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)
        
        log_line = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Step: {state.global_step} | "
            f"Epoch: {epoch:.2f} | Loss: {loss_str} | Eval Loss: {eval_loss_str} | "
            f"LR: {lr_str} | GPU Mem: {gpu_mem:.2f} GB | Tokens/sec: {tokens_sec}"
        )
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
        print(log_line)


def main():
    parser = argparse.ArgumentParser(description="QLoRA Dataset Fine-Tuning Pipeline")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config yaml file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    model_config = config["model"]
    quant_config = config["quantization"]
    lora_config = config["lora"]
    train_config = config["training"]
    data_config = config["data"]

    # --- SAFETY CHECKS ---
    has_cuda = check_cuda()
    check_disk_space(train_config["output_dir"], data_config.get("min_free_disk_gb", 5.0))
    train_raw = validate_dataset(data_config["train_path"], "train")
    eval_raw = validate_dataset(data_config["eval_path"], "eval")

    # Set seeds
    set_seed(train_config.get("seed", 42))

    # Load Tokenizer
    print(f"Loading tokenizer: {model_config['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], trust_remote_code=False)
    if tokenizer is None:
        raise RuntimeError("Safety Check Failure: Tokenizer could not be loaded.")
    
    # Configure padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Fallback Chat Template Setup
    if tokenizer.chat_template is None:
        print("Tokenizer has no default chat template. Configuring internal fallback template...")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
            "{% endif %}"
        )

    # Format Datasets
    def apply_template(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    train_dataset = Dataset.from_list(train_raw).map(apply_template, batched=True)
    eval_dataset = Dataset.from_list(eval_raw).map(apply_template, batched=True)

    # Load Base Model
    if has_cuda:
        # Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=get_dtype(quant_config["bnb_4bit_compute_dtype"]),
        )
        print(f"Loading quantized model on GPU: {model_config['name']}")
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=get_dtype(model_config["torch_dtype"]),
            attn_implementation=model_config.get("attn_implementation", "eager"),
            trust_remote_code=False,
        )
        model.config.use_cache = model_config.get("use_cache", False)

        # Prepare for training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_config.get("gradient_checkpointing", True)
        )
        if train_config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
    else:
        print(f"Loading unquantized model on CPU: {model_config['name']}")
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            device_map=None,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            trust_remote_code=False,
        )
        model.config.use_cache = model_config.get("use_cache", False)

    # LoRA Setup (Auto-detection of Linear Modules)
    target_modules = find_all_linear_names(model)
    print(f"Auto-detected Target Modules: {target_modules}")

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=target_modules,
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )

    model = get_peft_model(model, peft_config)
    
    # Calculate trainable parameters
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"\n--- Model Parameters Summary ---")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters:     {all_params:,}")
    print(f"Trainable %:          {100 * trainable_params / all_params:.4f}%")
    print(f"--------------------------------\n")

    # Logging Callback Setup
    logger_callback = TrainingLoggerCallback(
        log_path="logs/train.log",
        batch_size=train_config["per_device_train_batch_size"],
        grad_accum=train_config["gradient_accumulation_steps"],
        max_seq_length=data_config["max_seq_length"]
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=train_config["weight_decay"],
        warmup_ratio=train_config["warmup_ratio"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        max_grad_norm=train_config["max_grad_norm"],
        seed=train_config.get("seed", 42),
        bf16=train_config.get("bf16", True) if has_cuda else False,
        fp16=train_config.get("fp16", False) if has_cuda else False,
        use_cpu=not has_cuda,
        gradient_checkpointing=train_config.get("gradient_checkpointing", True) if has_cuda else False,
        
        # Strategies
        save_strategy=train_config["save_strategy"],
        save_steps=train_config["save_steps"],
        eval_strategy=train_config["eval_strategy"],
        eval_steps=train_config["eval_steps"],
        logging_strategy=train_config["logging_strategy"],
        logging_steps=train_config["logging_steps"],
        
        # Resume & best model settings
        load_best_model_at_end=train_config["load_best_model_at_end"],
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=train_config["greater_is_better"],
        report_to=train_config["report_to"],
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=data_config["max_seq_length"],
        callbacks=[logger_callback],
    )

    # Automatic Resume Setup
    latest_checkpoint = find_latest_checkpoint(train_config["output_dir"])
    if latest_checkpoint:
        print(f"Resuming training dynamically from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("No prior checkpoint discovered. Commencing fine-tuning from scratch.")
        trainer.train()

    # Save final model state and artifacts
    print(f"Saving final adapter and tokenizer to: {train_config['final_adapter_dir']}")
    trainer.save_model(train_config["final_adapter_dir"])
    tokenizer.save_pretrained(train_config["final_adapter_dir"])

    # Save configs for reproduction
    shutil.copy(config_path, os.path.join(train_config["final_adapter_dir"], "train_config.yaml"))
    
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
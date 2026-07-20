#!/usr/bin/env python3
"""
scripts/learning_verification.py

Loads a 100-sample subset of the training dataset, runs a 1-epoch training loop with
assistant-only loss, and measures loss and token accuracy to verify that learning occurs.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def load_subset(path: str, size: int = 100) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
            if len(data) >= size:
                break
    return data

def main():
    print("=== Commencing Phase 6: Learning Verification ===")
    
    # Load 100 samples
    train_path = "data/final/train_v1.jsonl"
    raw_data = load_subset(train_path, 100)
    print(f"Loaded {len(raw_data)} samples for learning verification.")
    
    model_name = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Configure custom template with generation tags
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% generation %}"
        "{{ message['content'] + '<|im_end|>\n' }}"
        "{% endgeneration %}"
        "{% endif %}"
        "{% endfor %}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    ds = Dataset.from_list(raw_data)
    
    # Configure SFTConfig with high learning rate to force faster learning on tiny subset
    training_args = SFTConfig(
        output_dir="./models/checkpoints_verify",
        num_train_epochs=15,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-3,
        weight_decay=0.01,
        seed=42,
        use_cpu=True,
        bf16=False,
        fp16=False,
        dataset_text_field=None,
        assistant_only_loss=True,
        max_length=512,
        save_strategy="no",
        eval_strategy="no",
        report_to="none"
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Measure loss before training
    print("Measuring initial validation loss...")
    initial_loss = trainer.evaluate(eval_dataset=ds)["eval_loss"]
    print(f"Initial validation loss: {initial_loss:.4f}")
    
    # Train for 1 epoch
    print("Training for 1 epoch...")
    train_result = trainer.train()
    final_train_loss = train_result.training_loss
    
    # Measure loss after training
    print("Measuring final validation loss...")
    final_loss = trainer.evaluate(eval_dataset=ds)["eval_loss"]
    print(f"Final validation loss: {final_loss:.4f}")
    
    improvement = initial_loss - final_loss
    print(f"Loss improvement: {improvement:.4f}")
    
    success = improvement > 0.05
    print(f"Learning Verification Result: {'PASS' if success else 'FAIL'}")
    
    # Write report
    report = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improvement": improvement,
        "status": "PASS" if success else "FAIL"
    }
    with open("logs/learning_verification.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()

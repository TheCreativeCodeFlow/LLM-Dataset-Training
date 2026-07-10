import os
import yaml
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/eval_config.yaml")

    model_config = config["model"]
    eval_config = config["eval"]
    gen_config = config["generation"]

    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=getattr(torch, model_config["torch_dtype"]),
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, model_config["adapter_path"])
    model.eval()

    dataset = load_dataset("json", data_files=eval_config["dataset_path"], split="train")
    if eval_config["max_samples"]:
        dataset = dataset.select(range(min(eval_config["max_samples"], len(dataset))))

    results = []
    for item in tqdm(dataset, desc="Evaluating"):
        if "messages" in item:
            # Filter to non-assistant messages to create the generation prompt
            generation_messages = [m for m in item["messages"] if m["role"] != "assistant"]
            prompt = tokenizer.apply_chat_template(
                generation_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            expected = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
        else:
            prompt = item.get("prompt", "")
            expected = item.get("solution", "")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=eval_config["max_seq_length"]).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config["max_new_tokens"],
                temperature=gen_config["temperature"],
                top_p=gen_config["top_p"],
                do_sample=gen_config["do_sample"],
                repetition_penalty=gen_config["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "expected": expected,
            "generated": generated.strip(),
        })

    output_path = "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Total samples: {len(results)}")


if __name__ == "__main__":
    main()
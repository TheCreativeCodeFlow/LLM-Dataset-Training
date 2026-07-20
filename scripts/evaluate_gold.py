#!/usr/bin/env python3
"""
scripts/evaluate_gold.py

Evaluates the base model or LoRA adapter on the curated gold set.
Generates evaluation results and human scoring sheets.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Bypassing large integer string conversion limits
sys.set_int_max_str_digits(0)


def load_gold_set(filepath: str) -> list:
    """Load the gold conversations from jsonl."""
    records = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on gold standard tutoring dataset.")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Base model identifier")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter weights")
    parser.add_argument("--gold-set", default="evaluation/gold_set/dsa_gold_v1.jsonl", help="Path to gold dataset")
    parser.add_argument("--dry-run", action="store_true", help="Perform setup and parse gold set without calling GPU model")
    args = parser.parse_args()

    # Load gold set
    gold_records = load_gold_set(args.gold_set)
    if not gold_records:
        raise FileNotFoundError(f"Gold evaluation set not found or empty: {args.gold_set}")

    print(f"Loaded {len(gold_records)} gold conversations from {args.gold_set}")

    results = []
    
    # Check dry run mode
    if args.dry_run:
        print("Dry-run active. Generating mocked responses for validation...")
        for r in gold_records:
            results.append({
                "id": r["id"],
                "topic": r["topic"],
                "conversation_type": r["conversation_type"],
                "difficulty": r["difficulty"],
                "prompt": r["messages"][-1]["content"],
                "response": "[DRY RUN MOCK RESPONSE] High-quality pedagogical tutoring explanation.",
                "scoring_rubric": r["scoring_rubric"]
            })
    else:
        # Import heavy ML libraries here
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"Loading base model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{'<|im_start|>assistant\n'}}"
                "{% endif %}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if args.adapter:
            print(f"Loading LoRA adapter from: {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)

        model.eval()

        print("Executing inference on gold set...")
        for idx, r in enumerate(gold_records):
            prompt_chat = r["messages"]
            formatted_prompt = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "id": r["id"],
                "topic": r["topic"],
                "conversation_type": r["conversation_type"],
                "difficulty": r["difficulty"],
                "prompt": r["messages"][-1]["content"],
                "response": response.strip(),
                "scoring_rubric": r["scoring_rubric"]
            })
            
            if (idx + 1) % 50 == 0:
                print(f"Evaluated {idx + 1}/{len(gold_records)} samples...")

    # Write output reports
    reports_dir = Path("evaluation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results_json = reports_dir / "gold_results.json"
    summary_md = reports_dir / "gold_summary.md"

    # Save gold_results.json
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save gold_summary.md (human evaluation scoring sheet)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Gold Evaluation Suite Summary & Human Scoring Sheet\n\n")
        f.write("Use this sheet to score model responses against rubrics.\n\n")
        f.write("## Scoring Rubric Definition\n")
        f.write("Score each metric on a scale of 1 (poor) to 5 (excellent):\n")
        f.write("1. **Technical Correctness** - Accuracy of the DSA explanation/code review.\n")
        f.write("2. **Educational Value** - Use of analogical reasoning and interactive guidance.\n")
        f.write("3. **Hint Quality** - Progressive scaffolding vs immediate answers.\n")
        f.write("4. **Beginner Friendliness** - Avoidance of dense academic jargon.\n")
        f.write("5. **Logical Consistency** - Cohesion and structured explanations.\n")
        f.write("6. **Solution Leakage** - Non-disclosure of core code until final prompts.\n")
        f.write("7. **Interview Usefulness** - Relevance and applicability to interviewing.\n\n")
        
        f.write("## Responses Review\n\n")
        for res in results[:20]:  # Output first 20 for preview readability in summary markdown
            f.write(f"### Sample ID: {res['id']} ({res['topic']} - {res['conversation_type']})\n")
            f.write(f"- **Difficulty:** {res['difficulty']}\n")
            f.write(f"- **User Input:** *{res['prompt']}*\n")
            f.write(f"- **Model Output:**\n```text\n{res['response']}\n```\n")
            f.write("- **Scoring Matrix:**\n")
            f.write("  | Technical | Educational | Hints | Friendliness | Consistency | Leakage | Interview |\n")
            f.write("  | --- | --- | --- | --- | --- | --- | --- |\n")
            f.write("  | [ ] /5 | [ ] /5 | [ ] /5 | [ ] /5 | [ ] /5 | [ ] /5 | [ ] /5 |\n\n")
            f.write("---\n\n")
            
        if len(results) > 20:
            f.write(f"\n*Note: Remaining {len(results) - 20} samples are stored in gold_results.json for programmatic parser loops.*\n")

    print(f"Gold evaluation reports successfully generated!")
    print(f"Results JSON: {results_json}")
    print(f"Scoring Sheet Summary: {summary_md}")


if __name__ == "__main__":
    main()

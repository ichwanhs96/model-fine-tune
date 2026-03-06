# /// script
# dependencies = ["transformers", "peft", "datasets", "lm-eval", "torch", "accelerate", "bitsandbytes"]
# ///

import os
import lm_eval

HF_TOKEN = os.environ.get("HF_TOKEN")

model_id   = "mistralai/Mistral-7B-v0.3"
adapter_id = "ichone/mistral-7b-medical-lora"

# Use bfloat16 — a 7B model fits in ~14GB on A10G (24GB VRAM)
# load_in_4bit is NOT passed here; lm_eval doesn't support it via model_args string
base_model_args = f"pretrained={model_id},dtype=bfloat16,token={HF_TOKEN}"
ft_model_args   = f"pretrained={model_id},peft={adapter_id},dtype=bfloat16,token={HF_TOKEN}"

TASKS = ["medqa_4options", "medmcqa"]

def run_benchmarks(model_args, label):
    print(f"\n--- Evaluating: {label} ---")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=TASKS,
        num_fewshot=0,
        batch_size=4,
    )
    for task, metrics in results["results"].items():
        acc = (
            metrics.get("acc,none")
            or metrics.get("acc_norm,none")
            or metrics.get("f1,none")
            or "N/A"
        )
        print(f"  {task}: {acc:.4f}" if isinstance(acc, float) else f"  {task}: {acc}")
    return results

base_results = run_benchmarks(base_model_args, "Base Model (Mistral-7B-v0.3)")
ft_results   = run_benchmarks(ft_model_args,   "Fine-tuned (medical-mistral-adapter)")

# Comparison table
print("\n=== Benchmark Comparison ===")
print(f"{'Task':<20} {'Base':>10} {'Fine-tuned':>12} {'Delta':>8}")
print("-" * 52)
for task in TASKS:
    def get_score(results, t=task):
        m = results["results"].get(t, {})
        return m.get("acc,none") or m.get("acc_norm,none") or m.get("f1,none")

    base_acc = get_score(base_results)
    ft_acc   = get_score(ft_results)
    if isinstance(base_acc, float) and isinstance(ft_acc, float):
        print(f"{task:<20} {base_acc:>10.4f} {ft_acc:>12.4f} {ft_acc - base_acc:>+8.4f}")

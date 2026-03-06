# /// script
# dependencies = ["transformers", "peft", "datasets", "trl", "bitsandbytes", "accelerate", "torch"]
# ///

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. Setup & Auth (HF Jobs automatically handles the token if passed as a secret)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Resume from latest Hub checkpoint if available
from huggingface_hub import list_repo_refs, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

RESUME_FROM_CHECKPOINT = None
ADAPTER_REPO = "ichone/mistral-7b-medical-lora"
try:
    refs = list_repo_refs(ADAPTER_REPO, token=HF_TOKEN)
    checkpoint_branches = sorted(
        [b.name for b in refs.branches if b.name.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    if checkpoint_branches:
        latest = checkpoint_branches[-1]
        print(f"Resuming from {latest}...")
        RESUME_FROM_CHECKPOINT = snapshot_download(
            ADAPTER_REPO, revision=latest, token=HF_TOKEN
        )
        print(f"Downloaded checkpoint to {RESUME_FROM_CHECKPOINT}")
except RepositoryNotFoundError:
    print("No existing checkpoint found, starting fresh.")

# 2. Dataset: Mix MedQuad (open-ended Q&A) + MedMCQA (MCQ) for better benchmark alignment
medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train[:8000]")
medmcqa = load_dataset("medmcqa", split="train[:4000]")

def format_medquad(example):
    return f"Question: {example['Question']}\nAnswer: {example['Answer']}</s>"

def format_medmcqa(example):
    options = f"A) {example['opa']}  B) {example['opb']}  C) {example['opc']}  D) {example['opd']}"
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    answer = answer_map.get(example["cop"], "A")
    return f"Question: {example['question']}\n{options}\nAnswer: {answer}) {example[f'op{answer.lower()}']}</s>"

from datasets import concatenate_datasets

medquad_formatted  = medquad.map(lambda x: {"text": format_medquad(x)},  remove_columns=medquad.column_names)
medmcqa_formatted  = medmcqa.map(lambda x: {"text": format_medmcqa(x)},  remove_columns=medmcqa.column_names)
dataset = concatenate_datasets([medquad_formatted, medmcqa_formatted]).shuffle(seed=42)

def format_base_style(example):
    return example["text"]

# 3. Model Loading (4-bit for efficiency)
model_id = "mistralai/Mistral-7B-v0.3" # THE BASE MODEL
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    token=HF_TOKEN
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# 4. LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Training Args (Cloud Optimized)
training_args = SFTConfig(
    output_dir="mistral-7b-medical-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_steps=500,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    push_to_hub=True,        # IMPORTANT: Uploads to your profile automatically
    hub_strategy="checkpoint",  # Push checkpoints every save_steps in case of timeout
    report_to="none",        # Can be "wandb" if you have an account
)

# 6. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=format_base_style,
    processing_class=tokenizer,
    args=training_args,
)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

# 7. Final Save & Push
trainer.push_to_hub("medical-mistral-adapter")
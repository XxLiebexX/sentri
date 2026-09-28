"""
LoRA fine-tune a small open code model on the vuln-detection dataset.

Designed to run on a single free-tier GPU (Colab T4 or Kaggle P100) --
LoRA only trains a small set of adapter weights, not the full model, so
it fits in ~16GB VRAM even for a 7B base model.

Run: python finetune/train.py
(after finetune/prepare_dataset.py has produced data/train.jsonl and data/val.jsonl)
"""

import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  # strong at code, fits LoRA on a free GPU
OUTPUT_DIR = "sentri-local-model"


def load_jsonl(path: str):
    return [json.loads(line) for line in open(path)]


def format_example(example: dict) -> str:
    """Turn a train.jsonl row into the chat-formatted training string."""
    return (
        f"<|im_start|>system\nYou are a code security reviewer.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    from datasets import Dataset

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,  # QLoRA -- keeps memory low enough for a free GPU
    )

    # LoRA config: only train small adapter matrices, not the full model.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check -- should be <1% of total params

    train_raw = load_jsonl("data/train.jsonl")
    val_raw = load_jsonl("data/val.jsonl")

    train_ds = Dataset.from_list([{"text": format_example(ex)} for ex in train_raw])
    val_ds = Dataset.from_list([{"text": format_example(ex)} for ex in val_raw])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch size 16, fits small VRAM
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

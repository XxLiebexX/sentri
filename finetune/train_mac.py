"""
LoRA fine-tune on Apple Silicon (M-series) using PyTorch's MPS backend.
"""

import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "sentri-local-model"


def load_jsonl(path: str):
    return [json.loads(line) for line in open(path)]


def format_example(example: dict) -> str:
    return (
        f"<|im_start|>system\nYou are a code security reviewer.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS not available -- check you're on Apple Silicon and have a "
            "recent enough PyTorch (`pip install --upgrade torch`)."
        )

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
    )
    model = model.to("mps")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_raw = load_jsonl("data/train.jsonl")
    val_raw = load_jsonl("data/val.jsonl")

    train_ds = Dataset.from_list([{"text": format_example(ex)} for ex in train_raw])
    val_ds = Dataset.from_list([{"text": format_example(ex)} for ex in val_raw])

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        bf16=False,
        report_to="none",
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting training on Apple Silicon (MPS)...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

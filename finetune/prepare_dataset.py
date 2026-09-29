"""
Prepare a fine-tuning dataset of (code, label) pairs for vulnerability/bug
detection. Uses public vulnerability datasets from Hugging Face so you don't
have to hand-label thousands of examples yourself.

Run this in Colab/Kaggle (needs internet access to huggingface.co, which
this sandbox doesn't have -- that's expected, not a bug).

Output format: JSONL, one example per line:
    {"code": "...", "label": "vulnerable" | "safe", "cwe": "CWE-89", "explanation": "..."}
"""

import json
import random
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_source_dataset():
    """
    Pulls a public vuln-detection dataset. A few good free options:
      - "PrimeVul" (function-level, vulnerable vs. fixed pairs)
      - "CVEfixes" (real CVE-linked commits, before/after code)
      - "bstee615/bigvul" (large, well-used baseline)

    Swap the dataset name below for whichever you want to try first --
    CVEfixes is the most "real-world" feeling for a portfolio project
    since every example traces back to an actual CVE.
    """
    from datasets import load_dataset  # pip install datasets

    ds = load_dataset("bstee615/bigvul", split="train")
    return ds


def build_training_examples(ds, max_examples: int = 4000):
    """
    Convert raw dataset rows into instruction-tuning format: the model
    sees code and is asked to classify + explain, matching the shape
    Stage 1 will actually query it with at inference time.
    """
    examples = []
    for row in ds:
        code = row.get("func_before") or row.get("code")
        if not code or len(code) > 6000:
            continue

        is_vuln = bool(row.get("vul", row.get("target", 0)))
        cwe = row.get("CWE_ID") or row.get("cwe") or "unknown"

        examples.append({
            "instruction": (
                "Review the following code for security vulnerabilities and "
                "bugs. Respond with a JSON object: "
                '{"verdict": "vulnerable"|"safe", "cwe": "...", "explanation": "..."}'
            ),
            "input": code,
            "output": json.dumps({
                "verdict": "vulnerable" if is_vuln else "safe",
                "cwe": cwe if is_vuln else "none",
                "explanation": (row.get("commit_message") or "")[:300] or "See CWE classification.",
            }),
        })

        if len(examples) >= max_examples:
            break

    return examples


def split_and_save(examples, val_ratio: float = 0.1):
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - val_ratio))
    train, val = examples[:split_idx], examples[split_idx:]

    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(OUTPUT_DIR / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train)} train examples, {len(val)} val examples to {OUTPUT_DIR}/")


if __name__ == "__main__":
    ds = load_source_dataset()
    examples = build_training_examples(ds)
    split_and_save(examples)

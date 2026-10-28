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


import os
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_ai_explanation(code: str, is_vuln: bool, cwe: str) -> str:
    """Uses Gemini to generate a high-quality review explanation for training."""
    import time
    import random
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
    
    if not os.environ.get("GEMINI_API_KEY"):
        return "See CWE classification."
        
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    
    condition = f"vulnerable to {cwe}" if is_vuln else "safe and secure"
    prompt = f"Write a clear, professional 2-sentence code review explanation of why the following code is {condition}. Do not use formatting, just return plain text.\n\n{code[:2000]}"
    
    # Exponential backoff
    base_wait = 2
    for attempt in range(5):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except (ResourceExhausted, DeadlineExceeded) as e:
            wait_time = base_wait * (2 ** attempt) + random.uniform(0, 1)
            print(f"API busy or timeout, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Gemini API error: {e}")
            break
            
    return "See CWE classification."

def build_and_save_training_examples(ds, max_examples: int = 4000, val_ratio: float = 0.1):
    """
    Convert raw dataset rows into instruction-tuning format and save incrementally.
    """
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    
    # We will just write everything to train.jsonl for now, 
    # or randomly decide to write to val.jsonl based on the ratio.
    import random
    random.seed(42)
    
    # Clear existing files if starting fresh
    with open(train_path, "w") as f: pass
    with open(val_path, "w") as f: pass

    print(f"Generating high-quality explanations for {max_examples} examples...")
    print(f"Examples will be saved incrementally. You can press Ctrl+C anytime and keep your progress!")
    
    count = 0
    train_count = 0
    val_count = 0
    
    for row in ds:
        code = row.get("func_before") or row.get("code")
        if not code or len(code) > 6000:
            continue

        is_vuln = bool(row.get("vul", row.get("target", 0)))
        cwe = row.get("CWE_ID") or row.get("cwe") or "unknown"
        
        # Call Gemini to write the explanation
        explanation = generate_ai_explanation(code, is_vuln, cwe)
        
        # If the API is completely exhausted and giving us fallbacks continuously, we might want to know,
        # but for now we just write whatever we get.

        example = {
            "instruction": (
                "Review the following code for security vulnerabilities and "
                "bugs. Respond with a JSON object: "
                '{"verdict": "vulnerable"|"safe", "cwe": "...", "explanation": "..."}'
            ),
            "input": code,
            "output": json.dumps({
                "verdict": "vulnerable" if is_vuln else "safe",
                "cwe": cwe if is_vuln else "none",
                "explanation": explanation,
            }),
        }

        # Save incrementally
        is_val = random.random() < val_ratio
        if is_val:
            with open(val_path, "a") as f:
                f.write(json.dumps(example) + "\n")
            val_count += 1
        else:
            with open(train_path, "a") as f:
                f.write(json.dumps(example) + "\n")
            train_count += 1
            
        count += 1

        if count % 10 == 0:
            print(f"Saved {count} / {max_examples} examples (Train: {train_count}, Val: {val_count})...")

        if count >= max_examples:
            break

    print(f"Finished! Wrote {train_count} train examples, {val_count} val examples.")


if __name__ == "__main__":
    ds = load_source_dataset()
    build_and_save_training_examples(ds)

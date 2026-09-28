"""
The number that matters for a resume: does local+escalation actually
save API calls without losing accuracy vs. calling Gemini on everything?

Run this against the held-out val.jsonl set and report, for each strategy:
  - accuracy (precision/recall on vulnerable vs safe)
  - % of examples that hit the Gemini API (cost proxy)

Run: python eval/compare_strategies.py
"""

import json
from src.local_model import LocalReviewer
from src.router import review_code, escalate_to_gemini


def load_val_set(path: str = "data/val.jsonl"):
    return [json.loads(line) for line in open(path)]


def score(predictions: list, ground_truth: list) -> dict:
    tp = fp = tn = fn = 0
    for pred, truth in zip(predictions, ground_truth):
        pred_positive = pred == "vulnerable"
        truth_positive = truth == "vulnerable"
        if pred_positive and truth_positive:
            tp += 1
        elif pred_positive and not truth_positive:
            fp += 1
        elif not pred_positive and truth_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def run_comparison():
    val_set = load_val_set()
    ground_truth = [json.loads(ex["output"])["verdict"] for ex in val_set]

    local_reviewer = LocalReviewer()

    local_only_preds, hybrid_preds, gemini_calls = [], [], 0

    for ex in val_set:
        code = ex["input"]

        # Strategy A: local model only, no escalation
        local_verdict = local_reviewer.review(code)
        local_only_preds.append(local_verdict.verdict)

        # Strategy B: local + escalation (the actual Sentri design)
        result = review_code(code, local_reviewer)
        hybrid_preds.append(result.verdict)
        if result.source == "gemini":
            gemini_calls += 1

    print("=== Local model only ===")
    print(score(local_only_preds, ground_truth))

    print("\n=== Local + Gemini escalation (Sentri design) ===")
    print(score(hybrid_preds, ground_truth))
    print(f"Escalated to Gemini: {gemini_calls}/{len(val_set)} "
          f"({100 * gemini_calls / len(val_set):.1f}% of calls)")

    print("\nThis last number is the pitch: if escalation rate is e.g. 30%, "
          "you're getting near-Gemini-only accuracy at ~30% of the API cost.")


if __name__ == "__main__":
    run_comparison()

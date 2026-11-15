"""
The number that matters for a resume: does local+escalation actually
save API calls without losing accuracy vs. calling Gemini on everything?

Run this against the held-out val.jsonl set and report, for each strategy:
  - accuracy (precision/recall on vulnerable vs safe)
  - % of examples that hit the Gemini API (cost proxy)

Run: python eval/compare_strategies.py [--max-samples N]
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Add project root to sys.path so imports work cleanly regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.local_model import LocalReviewer
from src.router import review_code, escalate_to_gemini, CONFIDENCE_THRESHOLD, ReviewResult


def load_val_set(path: str = "data/val.jsonl"):
    return [json.loads(line) for line in open(path) if line.strip()]


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

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "total": len(predictions),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


async def run_comparison(val_path: str = "data/val.jsonl", max_samples: int | None = None):
    val_set = load_val_set(val_path)
    if max_samples:
        val_set = val_set[:max_samples]

    if not val_set:
        print(f"[eval] No validation samples found in {val_path}")
        return

    ground_truth = []
    for ex in val_set:
        try:
            parsed = json.loads(ex["output"])
            ground_truth.append(parsed.get("verdict", "unknown"))
        except Exception:
            ground_truth.append("unknown")

    print(f"[eval] Initializing LocalReviewer (mps/cpu)...")
    local_reviewer = LocalReviewer()

    local_only_preds = []
    hybrid_preds = []
    gemini_calls = 0

    print(f"[eval] Running evaluation on {len(val_set)} samples...\n")

    for i, ex in enumerate(val_set, 1):
        code = ex["input"]
        truth = ground_truth[i - 1]
        print(f"--- Sample {i}/{len(val_set)} (Ground truth: {truth}) ---")

        # Strategy A: local model only, no escalation
        local_verdict = local_reviewer.review(code)
        local_only_preds.append(local_verdict.verdict)
        print(f"  [Local Model] verdict: {local_verdict.verdict} | confidence: {local_verdict.confidence:.3f}")

        # Strategy B: local + escalation (reuse local_verdict to avoid duplicate local inference)
        should_escalate = (
            local_verdict.confidence < CONFIDENCE_THRESHOLD
            or local_verdict.verdict in ("vulnerable", "unknown")
        )
        if should_escalate:
            result = await escalate_to_gemini(code)
        else:
            result = ReviewResult(
                verdict=local_verdict.verdict,
                cwe=local_verdict.cwe,
                explanation=local_verdict.explanation,
                source="local",
            )
        hybrid_preds.append(result.verdict)
        escalated = result.source == "gemini"
        if escalated:
            gemini_calls += 1
        print(f"  [Sentri Hybrid] verdict: {result.verdict} | source: {result.source} | escalated: {escalated}")

    print("\n" + "=" * 60)
    print("=== Strategy A: Local Model Only ===")
    print(json.dumps(score(local_only_preds, ground_truth), indent=2))

    print("\n=== Strategy B: Local + Gemini Escalation (Sentri Design) ===")
    print(json.dumps(score(hybrid_preds, ground_truth), indent=2))

    escalation_rate = (gemini_calls / len(val_set)) * 100
    print(f"\nEscalated to Gemini: {gemini_calls}/{len(val_set)} ({escalation_rate:.1f}% of calls)")
    print(f"API Cost Savings: {100.0 - escalation_rate:.1f}% reduction in cloud API calls vs. full cloud review")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sentri routing strategies against validation set.")
    parser.add_argument("--val-set", default="data/val.jsonl", help="Path to val.jsonl dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples to evaluate")
    args = parser.parse_args()

    asyncio.run(run_comparison(val_path=args.val_set, max_samples=args.max_samples))


if __name__ == "__main__":
    main()

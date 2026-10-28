"""
Loads the fine-tuned local model and runs inference. Kept separate from
the router so you can swap the local model out (e.g. quantized GGUF via
Ollama instead of raw transformers) without touching routing logic.
"""

import json
import threading
from dataclasses import dataclass


@dataclass
class LocalVerdict:
    verdict: str          # "vulnerable" | "safe"
    cwe: str
    explanation: str
    confidence: float     # 0-1, derived from output token probabilities


class LocalReviewer:
    _lock = threading.Lock()

    def __init__(self, model_path: str = "sentri-local-model", base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
        base = base.to(device)
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()

    def review(self, code: str) -> LocalVerdict:
        with self._lock:
            import torch

            prompt = (
                f"<|im_start|>system\nYou are a code security reviewer.<|im_end|>\n"
                f"<|im_start|>user\nReview the following code for security "
                f'vulnerabilities and bugs. Respond with a JSON object: '
                f'{{"verdict": "vulnerable"|"safe", "cwe": "...", "explanation": "..."}}'
                f"\n\n{code}<|im_end|>\n<|im_start|>assistant\n"
            )
            print("[local_model] Tokenizing prompt...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            print("[local_model] Generating...")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,          # deterministic -- we want consistent reviews
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            text = self.tokenizer.decode(output.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Confidence proxy: We care most about the model's confidence in its *verdict*,
            # which appears at the beginning of the generated JSON. We average the probabilities
            # of the first 20 tokens (which cover `{"verdict": "vulnerable/safe"}`) to get a
            # much more accurate representation of its certainty than the full explanation.
            probs = [torch.softmax(s, dim=-1).max().item() for s in output.scores]
            verdict_probs = probs[:20] if len(probs) >= 20 else probs
            confidence = sum(verdict_probs) / len(verdict_probs) if verdict_probs else 0.0

            import re
            try:
                match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
                json_str = match.group(0) if match else text.strip()
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # Malformed output is itself a strong signal to escalate.
                return LocalVerdict(verdict="unknown", cwe="unknown", explanation=text, confidence=0.0)

            return LocalVerdict(
                verdict=parsed.get("verdict", "unknown"),
                cwe=parsed.get("cwe", "unknown"),
                explanation=parsed.get("explanation", ""),
                confidence=confidence,
            )

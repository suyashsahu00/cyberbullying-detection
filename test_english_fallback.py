"""
Test Script: Verification of Bug 2 Fix on Known English False-Negative Examples.
Demonstrates Before (Raw MuRIL) vs After (MuRIL + English Hybrid Fallback).
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import classify_probabilities

def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 105)
    print(" 🧪 BUG 2 VERIFICATION TEST: ENGLISH FALSE-NEGATIVE RECOVERY")
    print("=" * 105)
    print("Comparing Raw MuRIL Model (BEFORE) vs MuRIL with English Hybrid Safety-Net (AFTER)\n")

    model_dir = os.path.join(ROOT_DIR, "models", "muril_cyberbullying_v2")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # Test cases: known English false-negatives / borderline slurs + benign controls
    test_cases = [
        {
            "id": 1,
            "text": "your mom is a whore",
            "expected_type": "Gender Harassment",
            "note": "Benchmark #3 (Raw model missed this with 50.5% Safe)"
        },
        {
            "id": 2,
            "text": "shut up you dumb bitch",
            "expected_type": "Gender Harassment",
            "note": "Short aggressive English misogynistic slur"
        },
        {
            "id": 3,
            "text": "go kill yourself loser kys",
            "expected_type": "Suicide Incitement",
            "note": "Severe targeted death / suicide threat"
        },
        {
            "id": 4,
            "text": "get out of here you dirty nigger",
            "expected_type": "Racial Slur",
            "note": "Severe racial slur attack"
        },
        {
            "id": 5,
            "text": "she only got the job because she is a thot and a whore",
            "expected_type": "Gender Harassment",
            "note": "Compound workplace misogyny"
        },
        {
            "id": 6,
            "text": "suck my dick motherfucker",
            "expected_type": "Harassment / Profanity",
            "note": "Severe English vulgar harassment"
        },
        {
            "id": 7,
            "text": "you are so helpful, thank you!",
            "expected_type": "BENIGN (Safe)",
            "note": "Control case: Must remain Safe (No False Positive)"
        },
        {
            "id": 8,
            "text": "Hope you have a wonderful and peaceful day ahead!",
            "expected_type": "BENIGN (Safe)",
            "note": "Control case: Must remain Safe (No False Positive)"
        }
    ]

    results = []

    for item in test_cases:
        text = item["text"]
        enc = tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=1).numpy()[0]

        # 1. BEFORE: Raw MuRIL (use_safety_net=False)
        raw_res = classify_probabilities(probs, id2label, method="two_stage", use_safety_net=False)

        # 2. AFTER: MuRIL with English Fallback (use_safety_net=True, passing raw_text)
        fixed_res = classify_probabilities(probs, id2label, method="two_stage", raw_text=text, use_safety_net=True)

        results.append({
            "id": item["id"],
            "text": text,
            "expected": item["expected_type"],
            "note": item["note"],
            "prob_safe": round(float(probs[3]) * 100, 1),
            "before_verdict": "Cyberbullying" if raw_res["is_cyberbullying"] else "Not Cyberbullying",
            "before_cat": raw_res["pred_class"],
            "before_conf": round(raw_res["confidence"], 1),
            "after_verdict": "Cyberbullying" if fixed_res["is_cyberbullying"] else "Not Cyberbullying",
            "after_cat": fixed_res["pred_class"],
            "after_conf": round(fixed_res["confidence"], 1),
            "safety_net_triggered": fixed_res["safety_net_triggered"]
        })

    # Print Detailed Execution Output
    for r in results:
        status_icon = "✅ FIXED" if (r["before_verdict"] == "Not Cyberbullying" and r["after_verdict"] == "Cyberbullying") else ("🟢 PRESERVED SAFE" if r["after_verdict"] == "Not Cyberbullying" else "✓ DETECTED")
        print(f"[{r['id']}] \"{r['text']}\"")
        print(f"    Target Type   : {r['expected']} ({r['note']})")
        print(f"    Raw Safe Prob : {r['prob_safe']}%")
        print(f"    BEFORE (Raw)  : Verdict = {r['before_verdict']:<18} | Category = {r['before_cat']:<18} | Conf = {r['before_conf']}%")
        print(f"    AFTER (Fixed) : Verdict = {r['after_verdict']:<18} | Category = {r['after_cat']:<18} | Conf = {r['after_conf']}% (SafetyNet={r['safety_net_triggered']})")
        print(f"    Status        : {status_icon}")
        print("-" * 105)

    # Summary Table
    print("\n" + "=" * 105)
    print(f"{'#':<3} | {'Input Text':<32} | {'BEFORE (Raw MuRIL)':<25} | {'AFTER (With English Fallback)':<28} | {'Impact':<10}")
    print("=" * 105)
    for r in results:
        short_text = (r['text'][:29] + "...") if len(r['text']) > 32 else r['text']
        before_str = f"{r['before_verdict']} ({r['before_cat']})"
        after_str = f"{r['after_verdict']} ({r['after_cat']})"
        impact = "RECOVERED" if (r['before_verdict'] == "Not Cyberbullying" and r['after_verdict'] == "Cyberbullying") else ("SAFE (OK)" if r['after_verdict'] == "Not Cyberbullying" else "HELD")
        print(f"{r['id']:<3} | {short_text:<32} | {before_str:<25} | {after_str:<28} | {impact:<10}")
    print("=" * 105)

if __name__ == "__main__":
    main()

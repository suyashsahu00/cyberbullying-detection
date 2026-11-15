import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = os.path.abspath('.')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import classify_probabilities

LABEL_MAP = {
    'age': 0,
    'ethnicity': 1,
    'gender': 2,
    'not_cyberbullying': 3,
    'other_cyberbullying': 4,
    'religion': 5
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]

text_19 = "Aam aadmi bina luck ke aage nahi badh sakte."

models = {
    "Archive": "models/muril_cyberbullying_v2_stable_archive_0e904b5",
    "Run 0 (s42)": "models/muril_exp_run0_control",
    "Run 0b (s43)": "models/muril_exp_run0b_seed43",
    "Ablation B (s42)": "models/muril_exp_ablation_B_all",
    "Ablation B (s43)": "models/muril_exp_ablation_B_seed43",
    "Ablation C (s42)": "models/muril_exp_ablation_C_oversample3x",
    "Ablation C (s43)": "models/muril_exp_ablation_C_seed43",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 90)
print(f"DETAILED ANALYSIS FOR SENTENCE #19: '{text_19}'")
print("=" * 90)

# 1. Full text prediction across models
for mname, mpath in models.items():
    tok = AutoTokenizer.from_pretrained(mpath)
    mod = AutoModelForSequenceClassification.from_pretrained(mpath).to(device)
    mod.eval()
    
    enc = tok([text_19], max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = mod(**enc).logits.float().cpu().numpy()[0]
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    
    prod = classify_probabilities(probs, ID_TO_LABEL, method='two_stage', raw_text=text_19, use_safety_net=True)
    arg_idx = int(np.argmax(probs))
    
    print(f"\n--- {mname} ---")
    print(f"  Production Verdict : {'Safe' if not prod['is_cyberbullying'] else 'BULLY'} (Confidence: {prod['confidence']:.2f}%)")
    print(f"  Argmax Verdict     : {ID_TO_LABEL[arg_idx]} (p={probs[arg_idx]:.4f})")
    print(f"  P(Safe / not_cyber): {probs[3]:.4f} | P(other_cyber): {probs[4]:.4f} | P(gender): {probs[2]:.4f} | P(age): {probs[0]:.4f}")
    print(f"  Full Probs: " + " | ".join([f"{c}: {probs[i]:.3f}" for i, c in enumerate(CLASS_NAMES)]))

# 2. Leave-one-out word ablation on Sentence 19
words = text_19.split()
print("\n" + "=" * 90)
print("LEAVE-ONE-OUT WORD ABLATION: P(Safe) AND P(other_cyberbullying)")
print("=" * 90)

sub_phrases = [text_19]
labels = ["FULL TEXT"]

for i in range(len(words)):
    omitted = words[i]
    phrase = " ".join([w for j, w in enumerate(words) if j != i])
    sub_phrases.append(phrase)
    labels.append(f"WITHOUT '{omitted}'")

# Also test specific phrase removals
sub_phrases.append("Aam aadmi aage nahi badh sakte.")
labels.append("WITHOUT 'bina luck ke'")
sub_phrases.append("bina luck ke aage nahi badh sakte.")
labels.append("WITHOUT 'Aam aadmi'")
sub_phrases.append("Aam aadmi mehnat ke aage badh sakte.")
labels.append("REPLACE with 'mehnat ke aage'")

for mname in ["Run 0 (s42)", "Run 0b (s43)", "Ablation B (s42)", "Ablation C (s42)", "Ablation B (s43)", "Ablation C (s43)"]:
    mpath = models[mname]
    tok = AutoTokenizer.from_pretrained(mpath)
    mod = AutoModelForSequenceClassification.from_pretrained(mpath).to(device)
    mod.eval()
    
    print(f"\n>>> Model: {mname}")
    print(f"{'Variation':<32} | {'Prod Verdict':<12} | {'P(Safe)':<10} | {'P(Other)':<10} | {'P(Age)':<10} | {'P(Gender)':<10}")
    print("-" * 88)
    
    with torch.no_grad():
        for lab, phrase in zip(labels, sub_phrases):
            enc = tok([phrase], max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = mod(**enc).logits.float().cpu().numpy()[0]
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            prod = classify_probabilities(probs, ID_TO_LABEL, method='two_stage', raw_text=phrase, use_safety_net=True)
            v_str = "Safe" if not prod['is_cyberbullying'] else f"BULLY ({prod['pred_class']})"
            print(f"{lab:<32} | {v_str:<12} | {probs[3]:>8.4f} | {probs[4]:>8.4f} | {probs[0]:>8.4f} | {probs[2]:>8.4f}")

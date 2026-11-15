import os
import sys
import json
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

def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p)) / n + (z**2) / (4 * (n**2)))) / denom
    lower = max(0.0, float(centre - margin))
    upper = min(1.0, float(centre + margin))
    return (lower, upper)

# Load holdout dataset - NO PRINTING ROWS OR TEXT
holdout_path = os.path.join(ROOT_DIR, 'data', 'final_holdout', 'final_human_test_v2.csv')
df = pd.read_csv(holdout_path)
total_samples = len(df)
y_true = df['category'].tolist()
texts = df['text'].tolist()

safe_mask = [y == 'not_cyberbullying' for y in y_true]
bully_mask = [y != 'not_cyberbullying' for y in y_true]
n_safe = sum(safe_mask)
n_bully = sum(bully_mask)

print(f"Loaded holdout set from: {holdout_path}")
print(f"Total samples: {total_samples} | Safe (not_cyberbullying): {n_safe} | Bully: {n_bully}")

models = {
    'Archive': 'models/muril_cyberbullying_v2_stable_archive_0e904b5',
    'Ablation B (s42)': 'models/muril_exp_ablation_B_all',
    'Ablation B (s43)': 'models/muril_exp_ablation_B_seed43'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference device: {device}")

results = {}

for mname, mpath in models.items():
    print(f"\nEvaluating {mname} ({mpath})...")
    tokenizer = AutoTokenizer.from_pretrained(mpath)
    model = AutoModelForSequenceClassification.from_pretrained(mpath)
    model.to(device)
    model.eval()

    # Inference batch
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    m_results = {}

    for mode in ['production', 'argmax']:
        preds = []
        for i in range(total_samples):
            p = probs[i]
            t = texts[i]
            if mode == 'production':
                res = classify_probabilities(p, ID_TO_LABEL, method='two_stage', safe_threshold=0.5, raw_text=t, use_safety_net=True)
                preds.append(res['pred_class'])
            else:
                preds.append(ID_TO_LABEL[int(np.argmax(p))])

        # Exact multi-class matches
        exact_correct = sum(1 for yt, yp in zip(y_true, preds) if yt == yp)
        acc = exact_correct / total_samples
        acc_ci = wilson_ci(exact_correct, total_samples)

        # Binary accuracy (Safe vs Bully)
        bin_correct = sum(1 for yt, yp in zip(y_true, preds) if (yt == 'not_cyberbullying') == (yp == 'not_cyberbullying'))
        bin_acc = bin_correct / total_samples
        bin_acc_ci = wilson_ci(bin_correct, total_samples)

        # Safe recall
        safe_correct = sum(1 for yt, yp in zip(y_true, preds) if yt == 'not_cyberbullying' and yp == 'not_cyberbullying')
        safe_rec = safe_correct / n_safe if n_safe > 0 else 0.0
        safe_rec_ci = wilson_ci(safe_correct, n_safe)

        # Bully recall (Binary: true bully predicted as ANY bully class)
        bully_detected = sum(1 for yt, yp in zip(y_true, preds) if yt != 'not_cyberbullying' and yp != 'not_cyberbullying')
        bully_rec = bully_detected / n_bully if n_bully > 0 else 0.0
        bully_rec_ci = wilson_ci(bully_detected, n_bully)

        # Bully exact multi-class recall (true bully predicted as exact bully class)
        bully_exact = sum(1 for yt, yp in zip(y_true, preds) if yt != 'not_cyberbullying' and yt == yp)
        bully_exact_rec = bully_exact / n_bully if n_bully > 0 else 0.0
        bully_exact_rec_ci = wilson_ci(bully_exact, n_bully)

        m_results[mode] = {
            'exact_accuracy': {
                'count': exact_correct,
                'total': total_samples,
                'value': acc,
                'wilson_95_ci': acc_ci
            },
            'binary_accuracy': {
                'count': bin_correct,
                'total': total_samples,
                'value': bin_acc,
                'wilson_95_ci': bin_acc_ci
            },
            'safe_recall': {
                'count': safe_correct,
                'total': n_safe,
                'value': safe_rec,
                'wilson_95_ci': safe_rec_ci
            },
            'bully_recall_binary': {
                'count': bully_detected,
                'total': n_bully,
                'value': bully_rec,
                'wilson_95_ci': bully_rec_ci
            },
            'bully_recall_exact': {
                'count': bully_exact,
                'total': n_bully,
                'value': bully_exact_rec,
                'wilson_95_ci': bully_exact_rec_ci
            }
        }

    results[mname] = m_results

# Save json
out_json = os.path.join(ROOT_DIR, 'scratch', 'final_holdout_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 110)
print(f"RAW FINAL HOLDOUT EVALUATION RESULTS (N={total_samples}, Safe={n_safe}, Bully={n_bully})")
print("=" * 110)
print(f"{'Model':<18} | {'Mode':<11} | {'Accuracy (Exact)':<22} | {'Safe Recall':<22} | {'Bully Recall (Binary)':<24} | {'Bully Recall (Exact)':<22}")
print("-" * 110)

for mname, m_res in results.items():
    for mode in ['production', 'argmax']:
        acc_obj = m_res[mode]['exact_accuracy']
        safe_obj = m_res[mode]['safe_recall']
        bully_obj = m_res[mode]['bully_recall_binary']
        bully_ex_obj = m_res[mode]['bully_recall_exact']

        acc_str = f"{acc_obj['count']}/{acc_obj['total']} ({acc_obj['value']*100:.2f}%) [{acc_obj['wilson_95_ci'][0]*100:.1f}%, {acc_obj['wilson_95_ci'][1]*100:.1f}%]"
        safe_str = f"{safe_obj['count']}/{safe_obj['total']} ({safe_obj['value']*100:.2f}%) [{safe_obj['wilson_95_ci'][0]*100:.1f}%, {safe_obj['wilson_95_ci'][1]*100:.1f}%]"
        bully_str = f"{bully_obj['count']}/{bully_obj['total']} ({bully_obj['value']*100:.2f}%) [{bully_obj['wilson_95_ci'][0]*100:.1f}%, {bully_obj['wilson_95_ci'][1]*100:.1f}%]"
        bully_ex_str = f"{bully_ex_obj['count']}/{bully_ex_obj['total']} ({bully_ex_obj['value']*100:.2f}%) [{bully_ex_obj['wilson_95_ci'][0]*100:.1f}%, {bully_ex_obj['wilson_95_ci'][1]*100:.1f}%]"

        print(f"{mname:<18} | {mode:<11} | {acc_str:<22} | {safe_str:<22} | {bully_str:<24} | {bully_ex_str:<22}")

print("=" * 110)
print(f"Results serialized to: {out_json}")

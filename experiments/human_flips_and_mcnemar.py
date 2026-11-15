import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from scipy.stats import binomtest
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

def evaluate_models(model_dirs, human_path='data/validation/hardneg_human_test_set.csv'):
    df = pd.read_csv(human_path)
    safe_df = df[df['category'] == 'not_cyberbullying'].copy().reset_index(drop=True)
    texts = safe_df['text'].tolist()
    langs = safe_df['language'].tolist()

    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for name, path in model_dirs.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, path {path} does not exist yet.")
            continue
        print(f"Evaluating {name} on {len(texts)} human safe sentences...")
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        mod.eval()

        preds_prod = []
        preds_argmax = []
        preds_t040 = []
        p_safe_vals = []

        with torch.no_grad():
            for t in texts:
                enc = tok([t], max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                logits = mod(**enc).logits.float()
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # 1. Production mode (with safety net)
                res_prod = classify_probabilities(probs, ID_TO_LABEL, method='two_stage', raw_text=t, use_safety_net=True)
                preds_prod.append(res_prod['is_cyberbullying'] == False) # True if Safe

                # 2. Argmax mode
                arg_cls = ID_TO_LABEL[int(np.argmax(probs))]
                preds_argmax.append(arg_cls == 'not_cyberbullying')

                # 3. Threshold 0.40 mode
                p_safe = float(probs[3])
                p_safe_vals.append(p_safe)
                preds_t040.append(p_safe >= 0.40)

        results[name] = {
            'prod_correct': preds_prod, # boolean list: True if predicted Safe
            'argmax_correct': preds_argmax,
            't040_correct': preds_t040,
            'p_safe': p_safe_vals
        }

    return safe_df, results

def mcnemar_exact(corr1, corr2):
    # corr1: baseline / control (boolean array)
    # corr2: treatment / ablation (boolean array)
    # a: both 1
    # b: 1 in corr1, 0 in corr2 (degraded)
    # c: 0 in corr1, 1 in corr2 (improved)
    # d: both 0
    a = sum(1 for c1, c2 in zip(corr1, corr2) if c1 and c2)
    b = sum(1 for c1, c2 in zip(corr1, corr2) if c1 and not c2)
    c = sum(1 for c1, c2 in zip(corr1, corr2) if not c1 and c2)
    d = sum(1 for c1, c2 in zip(corr1, corr2) if not c1 and not c2)
    n = b + c
    if n == 0:
        p_val = 1.0
    else:
        res = binomtest(b, n, 0.5, alternative='two-sided')
        p_val = res.pvalue
    return {'a': a, 'b': b, 'c': c, 'd': d, 'n_discordant': n, 'p_value': p_val}

def main():
    model_dirs = {
        'Archive': 'models/muril_cyberbullying_v2_stable_archive_0e904b5',
        'Run 0 (Seed 42)': 'models/muril_exp_run0_control',
        'Run 0b (Seed 43)': 'models/muril_exp_run0b_seed43',
        'Ablation B (Seed 42)': 'models/muril_exp_ablation_B_all',
        'Ablation B (Seed 43)': 'models/muril_exp_ablation_B_seed43',
        'Ablation C (Seed 42)': 'models/muril_exp_ablation_C_oversample3x',
        'Ablation C (Seed 43)': 'models/muril_exp_ablation_C_seed43'
    }

    safe_df, results = evaluate_models(model_dirs)

    # Let's save raw predictions
    out_dict = {'texts': safe_df['text'].tolist(), 'languages': safe_df['language'].tolist()}
    for m, d in results.items():
        out_dict[f"{m}_prod_safe"] = [bool(x) for x in d['prod_correct']]
        out_dict[f"{m}_p_safe"] = [round(float(x), 4) for x in d['p_safe']]

    with open('scratch/human_flips_raw.json', 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print("HUMAN TEST SET (30 SAFE SENTENCES) - PER-SENTENCE FLIP TABLE (PRODUCTION MODE)")
    print("=" * 95)

    headers = ["ID", "Lang", "Sentence", "Run0", "Run0b", "B(s42)", "C(s42)"]
    if 'Ablation B (Seed 43)' in results:
        headers.append("B(s43)")
    if 'Ablation C (Seed 43)' in results:
        headers.append("C(s43)")
    print(f"{headers[0]:<4} | {headers[1]:<8} | {headers[2]:<40} | " + " | ".join([f"{h:<7}" for h in headers[3:]]))
    print("-" * 115)

    for i in range(len(safe_df)):
        lang = safe_df.loc[i, 'language']
        text = safe_df.loc[i, 'text']
        t_trunc = (text[:37] + '...') if len(text) > 40 else text
        
        row_str = [f"{i+1:<4}", f"{lang:<8}", f"{t_trunc:<40}"]
        for h in headers[3:]:
            # find model key
            for k in results.keys():
                if h in k or (h == "Run0" and k == "Run 0 (Seed 42)") or (h == "Run0b" and k == "Run 0b (Seed 43)") or (h == "B(s42)" and k == "Ablation B (Seed 42)") or (h == "C(s42)" and k == "Ablation C (Seed 42)") or (h == "B(s43)" and k == "Ablation B (Seed 43)") or (h == "C(s43)" and k == "Ablation C (Seed 43)"):
                    is_safe = results[k]['prod_correct'][i]
                    sym = "Safe" if is_safe else "BULLY"
                    row_str.append(f"{sym:<7}")
                    break
        print(" | ".join(row_str))

    # McNemar tests
    print("\n" + "=" * 95)
    print("EXACT MCNEMAR TESTS (Safe Sentences N=30, Production Mode)")
    print("=" * 95)
    
    comparisons = [
        ('Ablation B (Seed 42)', 'Run 0 (Seed 42)'),
        ('Ablation B (Seed 42)', 'Run 0b (Seed 43)'),
        ('Ablation C (Seed 42)', 'Run 0 (Seed 42)'),
        ('Ablation C (Seed 42)', 'Run 0b (Seed 43)')
    ]
    if 'Ablation B (Seed 43)' in results:
        comparisons.extend([
            ('Ablation B (Seed 43)', 'Run 0 (Seed 42)'),
            ('Ablation B (Seed 43)', 'Run 0b (Seed 43)')
        ])
    if 'Ablation C (Seed 43)' in results:
        comparisons.extend([
            ('Ablation C (Seed 43)', 'Run 0 (Seed 42)'),
            ('Ablation C (Seed 43)', 'Run 0b (Seed 43)')
        ])

    mcn_results = {}
    for treat, ctrl in comparisons:
        if treat not in results or ctrl not in results:
            continue
        m = mcnemar_exact(results[ctrl]['prod_correct'], results[treat]['prod_correct'])
        mcn_results[f"{treat} vs {ctrl}"] = m
        print(f"\n--- {treat} vs {ctrl} ---")
        print(f"  Contingency Table: Both Correct (a) = {m['a']}, Degraded in Treatment (b) = {m['b']}, Improved in Treatment (c) = {m['c']}, Both Incorrect (d) = {m['d']}")
        print(f"  Discordant Pairs (b + c) = {m['n_discordant']}")
        print(f"  Exact Two-Tailed Binomial McNemar p-value: {m['p_value']:.5f} (p = {m['p_value']})")

    with open('scratch/mcnemar_results.json', 'w', encoding='utf-8') as f:
        json.dump(mcn_results, f, indent=2)

if __name__ == '__main__':
    main()


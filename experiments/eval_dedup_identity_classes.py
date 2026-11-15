import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
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
IDENTITY_CLASSES = ['age', 'ethnicity', 'gender', 'religion']

df_dedup = pd.read_parquet('data/processed/combined_test_dedup.parquet')
col_text = 'cleaned_text' if 'cleaned_text' in df_dedup.columns else 'tweet_text'
col_label = 'cyberbullying_type' if 'cyberbullying_type' in df_dedup.columns else 'category'
texts = df_dedup[col_text].astype(str).tolist()
y_true = df_dedup[col_label].tolist()

models = {
    'Archive': 'models/muril_cyberbullying_v2_stable_archive_0e904b5',
    'Run 0 (s42)': 'models/muril_exp_run0_control',
    'Run 0b (s43)': 'models/muril_exp_run0b_seed43',
    'Ablation B (s42)': 'models/muril_exp_ablation_B_all',
    'Ablation B (s43)': 'models/muril_exp_ablation_B_seed43',
    'Ablation C (s42)': 'models/muril_exp_ablation_C_oversample3x',
    'Ablation C (s43)': 'models/muril_exp_ablation_C_seed43'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_metrics = {}

for mname, mpath in models.items():
    print(f"Evaluating {mname} on {len(texts)} dedup blind test samples...")
    tok = AutoTokenizer.from_pretrained(mpath)
    mod = AutoModelForSequenceClassification.from_pretrained(mpath).to(device)
    mod.eval()
    
    # Batch predict probs
    all_probs = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i+batch_size]
            enc = tok(b_texts, max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
            p = torch.softmax(mod(**enc).logits.float(), dim=1).cpu().numpy()
            all_probs.extend(p)
            
    all_probs = np.array(all_probs)
    
    # 1. Production Mode
    prod_preds = []
    for i in range(len(texts)):
        res = classify_probabilities(all_probs[i], ID_TO_LABEL, method='two_stage', raw_text=texts[i], use_safety_net=True)
        prod_preds.append(res['pred_class'])
    
    # 2. Argmax Mode
    argmax_preds = [ID_TO_LABEL[int(np.argmax(p))] for p in all_probs]
    
    rep_prod = classification_report(y_true, prod_preds, output_dict=True, zero_division=0)
    rep_arg = classification_report(y_true, argmax_preds, output_dict=True, zero_division=0)
    
    out_metrics[mname] = {
        'production': {c: rep_prod[c] for c in IDENTITY_CLASSES},
        'argmax': {c: rep_arg[c] for c in IDENTITY_CLASSES}
    }

print("\n" + "=" * 115)
print("DEDUPLICATED BLIND TEST (N=4,963): IDENTITY CLASSES (Age, Ethnicity, Gender, Religion)")
print("=" * 115)

for mode in ['production', 'argmax']:
    print(f"\n### DECISION MODE: {mode.upper()}")
    for cls in IDENTITY_CLASSES:
        print(f"\n--- Class: {cls.upper()} ---")
        print(f"{'Model':<24} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
        print("-" * 75)
        for mname in models.keys():
            m_stat = out_metrics[mname][mode][cls]
            p_val = m_stat['precision'] * 100.0
            r_val = m_stat['recall'] * 100.0
            f1_val = m_stat['f1-score'] * 100.0
            sup = int(m_stat['support'])
            print(f"{mname:<24} | {p_val:>10.2f}% | {r_val:>10.2f}% | {f1_val:>10.2f}% | {sup:>8}")

with open('scratch/dedup_identity_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(out_metrics, f, indent=2)

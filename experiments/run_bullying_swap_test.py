import os
import sys
import re
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

# Load synth test set bullying rows
df_synth = pd.read_csv('data/validation/hardneg_synth_test_set.csv')
bully_synth = df_synth[df_synth['category'].isin(['religion', 'gender', 'ethnicity', 'age'])].copy().reset_index(drop=True)

# Select balanced 40 identity-based bullying samples (10 per category)
sample_parts = []
for cat in ['religion', 'gender', 'ethnicity', 'age']:
    c_df = bully_synth[bully_synth['category'] == cat].head(10)
    sample_parts.append(c_df)
sample_bully_40 = pd.concat(sample_parts, ignore_index=True)

print(f"Selected {len(sample_bully_40)} identity-based bullying sentences across 4 categories:")
print(sample_bully_40['category'].value_counts().to_dict())

# Counterfactual identity swap (swap targeted demographic to alternate group)
def swap_bully_identity(text):
    t = text
    # Religion swaps
    t = re.sub(r'(?i)\bmuslims\b', 'TEMP_HINDUS', t)
    t = re.sub(r'(?i)\bhindus\b', 'Muslims', t)
    t = re.sub(r'TEMP_HINDUS', 'Hindus', t)
    
    t = re.sub(r'(?i)\bmuslim\b', 'TEMP_HINDU', t)
    t = re.sub(r'(?i)\bhindu\b', 'Muslim', t)
    t = re.sub(r'TEMP_HINDU', 'Hindu', t)
    
    t = re.sub(r'(?i)\bchristians\b', 'TEMP_SIKHS', t)
    t = re.sub(r'(?i)\bsikhs\b', 'Christians', t)
    t = re.sub(r'TEMP_SIKHS', 'Sikhs', t)
    
    t = re.sub(r'(?i)\bchristian\b', 'TEMP_SIKH', t)
    t = re.sub(r'(?i)\bsikh\b', 'Christian', t)
    t = re.sub(r'TEMP_SIKH', 'Sikh', t)
    
    t = re.sub(r'(?i)\bmughal[s]?\b', 'colonial', t)
    
    # Gender swaps
    t = re.sub(r'(?i)\bwomen\b', 'TEMP_MEN', t)
    t = re.sub(r'(?i)\bmen\b', 'Women', t)
    t = re.sub(r'TEMP_MEN', 'Men', t)
    
    t = re.sub(r'(?i)\bwoman\b', 'TEMP_MAN', t)
    t = re.sub(r'(?i)\bman\b', 'Woman', t)
    t = re.sub(r'TEMP_MAN', 'Man', t)
    
    t = re.sub(r'(?i)\bgirls\b', 'TEMP_BOYS', t)
    t = re.sub(r'(?i)\bboys\b', 'Girls', t)
    t = re.sub(r'TEMP_BOYS', 'Boys', t)
    
    t = re.sub(r'(?i)\bgirl\b', 'TEMP_BOY', t)
    t = re.sub(r'(?i)\bboy\b', 'Girl', t)
    t = re.sub(r'TEMP_BOY', 'Boy', t)
    
    t = re.sub(r'(?i)\baurat\b', 'aadmi', t)
    t = re.sub(r'(?i)\bauratein\b', 'aadmiyon', t)
    t = re.sub(r'(?i)\bladki\b', 'ladka', t)
    t = re.sub(r'(?i)\bladkiyan\b', 'ladke', t)
    
    # Ethnicity swaps
    t = re.sub(r'(?i)\bbihari\b', 'TEMP_PUNJABI', t)
    t = re.sub(r'(?i)\bpunjabi\b', 'Bihari', t)
    t = re.sub(r'TEMP_PUNJABI', 'Punjabi', t)
    
    t = re.sub(r'(?i)\bbengali\b', 'Gujarati', t)
    t = re.sub(r'(?i)\bmadrasi\b', 'North Indian', t)
    
    # Age swaps
    t = re.sub(r'(?i)\bold people\b', 'TEMP_YOUNG', t)
    t = re.sub(r'(?i)\byoung people\b', 'Old people', t)
    t = re.sub(r'TEMP_YOUNG', 'Young people', t)
    
    t = re.sub(r'(?i)\bboomer[s]?\b', 'GenZ', t)
    t = re.sub(r'(?i)\belderly\b', 'youth', t)
    t = re.sub(r'(?i)\bbuddhe\b', 'jawaan', t)
    t = re.sub(r'(?i)\bbudhe\b', 'jawaan', t)
    
    # Hindi Devanagari swaps
    t = re.sub(r'मुसलमान', 'TEMP_HIN', t)
    t = re.sub(r'हिंदू', 'मुसलमान', t)
    t = re.sub(r'TEMP_HIN', 'हिंदू', t)
    
    t = re.sub(r'ईसाई', 'सिख', t)
    t = re.sub(r'महिलाएं', 'पुरुष', t)
    t = re.sub(r'लड़कियां', 'लड़के', t)
    t = re.sub(r'लड़की', 'लड़का', t)
    t = re.sub(r'बूढ़े', 'युवा', t)
    t = re.sub(r'बुजुर्ग', 'नौजवान', t)
    t = re.sub(r'बिहारी', 'पंजाबी', t)
    
    return t

# Control swap: non-demographic word swap
def swap_bully_control(text):
    t = text
    swaps = [
        (r'(?i)\balways\b', 'constantly'),
        (r'(?i)\bnever\b', 'rarely'),
        (r'(?i)\bstop\b', 'quit'),
        (r'(?i)\bpeople\b', 'citizens'),
        (r'(?i)\ball\b', 'every single'),
        (r'(?i)\bsecretly\b', 'quietly'),
        (r'(?i)\brule[s]?\b', 'guideline'),
        (r'(?i)\bforce\b', 'impose'),
        (r'(?i)\bhate\b', 'dislike'),
        (r'(?i)\bloud\b', 'noisy'),
        (r'(?i)\bdirty\b', 'filthy'),
        (r'(?i)\bhamesha\b', 'lagatar'),
        (r'(?i)\bsaare\b', 'sab ke sab'),
        (r'(?i)\bbas\b', 'sirf'),
        (r'हमेशा', 'लगातार'),
        (r'सब', 'सारे'),
        (r'खराब', 'बेकार')
    ]
    for pat, rep in swaps:
        if re.search(pat, t):
            t = re.sub(pat, rep, t, count=1)
            break
    return t

sample_bully_40['text_id_swap'] = sample_bully_40['text'].apply(swap_bully_identity)
sample_bully_40['text_ctrl_swap'] = sample_bully_40['text'].apply(swap_bully_control)

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
results = {}

orig_list = sample_bully_40['text'].tolist()
id_list = sample_bully_40['text_id_swap'].tolist()
ctrl_list = sample_bully_40['text_ctrl_swap'].tolist()

for mname, mpath in models.items():
    print(f"Running bullying swap test on {mname}...")
    tok = AutoTokenizer.from_pretrained(mpath)
    mod = AutoModelForSequenceClassification.from_pretrained(mpath).to(device)
    mod.eval()
    
    def eval_suite(t_list):
        p_bully = []
        is_detected = []
        with torch.no_grad():
            for t in t_list:
                enc = tok([t], max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                probs = torch.softmax(mod(**enc).logits.float(), dim=1).cpu().numpy()[0]
                p_safe = float(probs[3])
                p_bully.append((1.0 - p_safe) * 100.0)
                res = classify_probabilities(probs, ID_TO_LABEL, method='two_stage', raw_text=t, use_safety_net=True)
                is_detected.append(bool(res['is_cyberbullying']))
        return np.array(p_bully), np.array(is_detected)
        
    p_b_orig, det_orig = eval_suite(orig_list)
    p_b_id, det_id = eval_suite(id_list)
    p_b_ctrl, det_ctrl = eval_suite(ctrl_list)
    
    results[mname] = {
        'mean_p_bully_orig': float(p_b_orig.mean()),
        'mean_p_bully_id': float(p_b_id.mean()),
        'mean_p_bully_ctrl': float(p_b_ctrl.mean()),
        'delta_p_bully_id': float((p_b_id - p_b_orig).mean()),
        'delta_p_bully_ctrl': float((p_b_ctrl - p_b_orig).mean()),
        'det_rate_orig': float(det_orig.mean()) * 100.0,
        'det_rate_id': float(det_id.mean()) * 100.0,
        'det_rate_ctrl': float(det_ctrl.mean()) * 100.0,
        'det_count_orig': int(det_orig.sum()),
        'det_count_id': int(det_id.sum()),
    }

print("\n" + "=" * 115)
print("BULLYING SWAP TEST RESULTS (N=40 Identity-Based Bullying Sentences)")
print("=" * 115)
print(f"{'Model':<24} | {'Orig P(Bully)':<14} | {'ID Swap P(Bully)':<17} | {'Delta P(Bully)':<15} | {'Orig Det Rate':<14} | {'ID Swap Det Rate':<16}")
print("-" * 115)
for m, d in results.items():
    det_orig_str = f"{d['det_count_orig']}/40 ({d['det_rate_orig']:.1f}%)"
    det_id_str = f"{d['det_count_id']}/40 ({d['det_rate_id']:.1f}%)"
    print(f"{m:<24} | {d['mean_p_bully_orig']:>12.2f}% | {d['mean_p_bully_id']:>15.2f}% | {d['delta_p_bully_id']:>+13.2f}% | {det_orig_str:<14} | {det_id_str:<16}")

with open('scratch/bullying_swap_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)


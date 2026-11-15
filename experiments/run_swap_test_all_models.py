import os
import sys
import torch
import numpy as np
import pandas as pd
import re
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = os.path.abspath('.')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load hardneg safe samples with identity terms
df_hn = pd.read_csv(os.path.join(ROOT_DIR, "data", "validation", "hardneg_synth_test_set.csv"))
safe_hn = df_hn[df_hn["category"] == "not_cyberbullying"].copy().reset_index(drop=True)

id_pattern = re.compile(
    r'(?<!\w)(mughal[s]?|मुग़ल|मुगल|muslim[s]?|musalman[s]?|मुस्लिम|मुसलमान|hindu[s]?|हिन्दू|हिंदू|sikh[s]?|सिख|christian[s]?|isaai|esai|ईसाई|bihari[s]?|बिहारी|madrasi[s]?|मद्रासी|women|woman|aurat|aurtein|महिला|महिलाएं|महिलाओ|औरत|औरतें|girl[s]?|ladki|ladkiyan|ladkiyon|लड़की|लड़कियां|लड़कियों|old\s+people|elderly|buddhe|budhe|बुड्ढे|बुजुर्ग|बूढ़े|बूढ़ा|boomer[s]?)(?!\w)',
    re.IGNORECASE
)

safe_hn["has_id"] = safe_hn["text"].fillna("").str.contains(id_pattern)
with_id_df = safe_hn[safe_hn["has_id"]].copy().reset_index(drop=True)
sample_40 = with_id_df.head(40).copy()

def natural_identity_swap(text):
    t = text
    t = re.sub(r'(?i)historic\s+mughal\s+gardens', 'historic botanical gardens', t)
    t = re.sub(r'(?i)mughal\s+gardens', 'botanical gardens', t)
    t = re.sub(r'(?i)mughal\s+architectural\s+monuments', 'medieval architectural monuments', t)
    t = re.sub(r'(?i)mughal\s+period', 'medieval period', t)
    t = re.sub(r'(?i)mughal\s+era', 'medieval era', t)
    t = re.sub(r'(?i)mughal\s+courts', 'royal courts', t)
    t = re.sub(r'(?i)mughal\s+fortress', 'historic fortress', t)
    t = re.sub(r'(?i)mughal\s+empire', 'regional kingdom', t)
    t = re.sub(r'(?i)mughal\s+history', 'regional history', t)
    t = re.sub(r'(?i)mughal\s+coinage', 'ancient coinage', t)
    t = re.sub(r'(?i)mughal', 'ancient', t)
    t = re.sub(r'(?i)christian\s+charity\s+trust', 'philanthropic trust', t)
    t = re.sub(r'(?i)christian', 'independent', t)
    t = re.sub(r'(?i)muslim\s+traders', 'local merchants', t)
    t = re.sub(r'(?i)muslim\s+padosiyon', 'colony padosiyon', t)
    t = re.sub(r'(?i)muslim', 'neighborhood', t)
    t = re.sub(r'(?i)hindu\s+youth\s+club', 'neighborhood youth club', t)
    t = re.sub(r'(?i)hindu', 'traditional', t)
    t = re.sub(r'(?i)sikh\s+volunteers', 'community volunteers', t)
    t = re.sub(r'(?i)sikh\s+community', 'neighborhood welfare group', t)
    t = re.sub(r'(?i)sikh', 'community', t)
    t = re.sub(r'(?i)bihari\s+workers', 'construction workers', t)
    t = re.sub(r'(?i)elderly\s+citizens', 'senior residents', t)
    t = re.sub(r'(?i)buddhe\s+aur\s+bache', 'buzurg aur bache', t)
    t = re.sub(r'(?i)every\s+woman\s+candidate', 'every recruit candidate', t)
    t = re.sub(r'(?i)girl\s+student', 'youth student', t)
    t = re.sub(r'हिंदू\s+और\s+मुस्लिम', 'सभी स्थानीय', t)
    t = re.sub(r'मुस्लिम', 'स्थानीय', t)
    t = re.sub(r'हिंदू', 'पारंपरिक', t)
    t = re.sub(r'ईसाई', 'परोपकारी', t)
    t = re.sub(r'सिख', 'सामुदायिक', t)
    t = re.sub(r'मुगल', 'ऐतिहासिक', t)
    t = re.sub(r'मुग़ल', 'ऐतिहासिक', t)
    t = re.sub(r'बुजुर्ग', 'वरिष्ठ', t)
    t = re.sub(r'बूढ़े', 'वरिष्ठ नागरिकों', t)
    t = re.sub(r'लड़कियां', 'छात्राएं', t)
    t = re.sub(r'महिलाएं', 'नागरिक', t)
    t = re.sub(r'aurat', 'mahila', t)
    return t

def control_swap(text):
    t = text
    swaps = [
        (r'(?i)\bvisitors\b', 'guests'),
        (r'(?i)\bopen\b', 'accessible'),
        (r'(?i)\bfestival\b', 'celebration'),
        (r'(?i)\bannual\b', 'yearly'),
        (r'(?i)\bweekly\b', 'regular'),
        (r'(?i)\bparticipated\b', 'took part'),
        (r'(?i)\bactive\b', 'energetic'),
        (r'(?i)\blocal\b', 'nearby'),
        (r'(?i)\bconducts\b', 'hosts'),
        (r'(?i)\blibrary\b', 'reading center'),
        (r'(?i)\bclasses\b', 'workshops'),
        (r'(?i)\bmedical\b', 'health'),
        (r'(?i)\bshelters\b', 'tents'),
        (r'(?i)\bhospital\b', 'clinic'),
        (r'(?i)\bdocumentary\b', 'feature film'),
        (r'(?i)\bscreening\b', 'presentation'),
        (r'(?i)\barchaeologists\b', 'researchers'),
        (r'(?i)\bteachers\b', 'instructors'),
        (r'(?i)\bparivar\b', 'khandan'),
        (r'(?i)\bkaam\b', 'sewa'),
        (r'(?i)\bbudget\b', 'funds'),
        (r'(?i)\bfree\b', 'complimentary'),
        (r'(?i)\bschool\b', 'academy'),
        (r'(?i)\bpadosiyon\b', 'neighbors'),
        (r'सहयोग', 'मदद'),
        (r'आयोजित', 'शुरू'),
        (r'तैयारी', 'व्यवस्था'),
        (r'वित्तीय', 'आर्थिक'),
        (r'समीक्षकों', 'आलोचकों'),
        (r'प्रशंसा', 'तारीफ'),
        (r'चर्चा', 'बातचीत')
    ]
    for pat, rep in swaps:
        if re.search(pat, t):
            t = re.sub(pat, rep, t, count=1)
            break
    return t

sample_40["text_id_swap"] = sample_40["text"].apply(natural_identity_swap)
sample_40["text_ctrl_swap"] = sample_40["text"].apply(control_swap)

def run_models_swap_test(model_dirs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    orig_list = sample_40["text"].tolist()
    id_list = sample_40["text_id_swap"].tolist()
    ctrl_list = sample_40["text_ctrl_swap"].tolist()

    for name, path in model_dirs.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, path {path} not found.")
            continue
        print(f"Running swap test on {name}...")
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        mod.eval()

        def get_p_safe(text_list):
            probs_out = []
            with torch.no_grad():
                for i in range(0, len(text_list), 64):
                    batch = text_list[i:i+64]
                    enc = tok(batch, max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                    probs = torch.softmax(mod(**enc).logits.float(), dim=1).cpu().numpy()
                    probs_out.extend(probs[:, 3] * 100.0)
            return np.array(probs_out)

        p_orig = get_p_safe(orig_list)
        p_id_swap = get_p_safe(id_list)
        p_ctrl_swap = get_p_safe(ctrl_list)

        delta_id = p_id_swap - p_orig
        delta_ctrl = p_ctrl_swap - p_orig

        results[name] = {
            "mean_p_orig": float(p_orig.mean()),
            "mean_p_id_swap": float(p_id_swap.mean()),
            "mean_p_ctrl_swap": float(p_ctrl_swap.mean()),
            "mean_delta_id": float(delta_id.mean()),
            "mean_delta_ctrl": float(delta_ctrl.mean()),
            "num_boosted_id": int((delta_id > 0).sum()),
            "num_boosted_ctrl": int((delta_ctrl > 0).sum()),
            "flipped_to_safe_id": int(((p_orig < 50.0) & (p_id_swap >= 50.0)).sum()),
            "flipped_to_safe_ctrl": int(((p_orig < 50.0) & (p_ctrl_swap >= 50.0)).sum())
        }

    return results

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

    results = run_models_swap_test(model_dirs)
    with open('scratch/swap_test_all_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 105)
    print("IDENTITY-SWAP TEST WITH CONTROL (N=40 Safe Sentences with Identity Terms)")
    print("=" * 105)
    print(f"{'Model':<24} | {'Orig P(Safe)':<13} | {'ID Swap P(Safe)':<16} | {'Delta(ID)':<11} | {'Ctrl P(Safe)':<14} | {'Delta(Ctrl)':<11}")
    print("-" * 105)
    for m, d in results.items():
        print(f"{m:<24} | {d['mean_p_orig']:>11.2f}% | {d['mean_p_id_swap']:>14.2f}% | {d['mean_delta_id']:>+9.2f}% | {d['mean_p_ctrl_swap']:>12.2f}% | {d['mean_delta_ctrl']:>+9.2f}%")

if __name__ == '__main__':
    main()


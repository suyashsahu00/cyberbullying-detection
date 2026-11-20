import os
import sys
import gc
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LABEL_MAP = {
    "age": 0,
    "ethnicity": 1,
    "gender": 2,
    "not_cyberbullying": 3,
    "other_cyberbullying": 4,
    "religion": 5
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
SAFE_IDX = LABEL_MAP["not_cyberbullying"]

MODELS = {
    "archive": os.path.join(ROOT_DIR, "models", "muril_cyberbullying_v2_stable_archive_0e904b5"),
    "run0": os.path.join(ROOT_DIR, "models", "muril_exp_run0_control"),
    "run0b": os.path.join(ROOT_DIR, "models", "muril_exp_run0b_seed43"),
    "b_s42": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_B_all"),
    "b_s43": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_B_seed43"),
    "c_s42": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_C_oversample3x"),
    "c_s43": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_C_seed43"),
}

def predict_probs(model, tokenizer, texts, device, batch_size=64, max_len=128):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if pd.notna(t) else "" for t in texts[i:i+batch_size]]
            enc = tokenizer(batch, max_length=max_len, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)

def compute_exact_mcnemar(correct_m1, correct_m2, name1="Model 1", name2="Model 2"):
    # correct_m1: bool array (e.g. Archive)
    # correct_m2: bool array (e.g. Ensemble A)
    # a: both correct
    # b: m1 correct, m2 incorrect (m1 wins)
    # c: m1 incorrect, m2 correct (m2 wins)
    # d: both incorrect
    a = int(np.sum(correct_m1 & correct_m2))
    b = int(np.sum(correct_m1 & ~correct_m2))
    c = int(np.sum(~correct_m1 & correct_m2))
    d = int(np.sum(~correct_m1 & ~correct_m2))
    n = b + c
    if n == 0:
        p_val = 1.0
    else:
        res = binomtest(b, n, 0.5, alternative="two-sided")
        p_val = float(res.pvalue)
    
    return {
        "comparison": f"{name2} vs {name1}",
        "both_correct_a": a,
        "m1_correct_m2_incorrect_b": b,
        "m1_incorrect_m2_correct_c": c,
        "both_incorrect_d": d,
        "discordant_pairs_n": n,
        "exact_p_value": p_val
    }

def bootstrap_ci(preds, y_true, safe_idx, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    acc_samples = []
    safe_rec_samples = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_preds = preds[idx]
        sample_y = y_true[idx]
        
        acc = np.mean(sample_preds == sample_y)
        acc_samples.append(acc)
        
        safe_mask = (sample_y == safe_idx)
        if np.sum(safe_mask) > 0:
            safe_rec = np.mean(sample_preds[safe_mask] == safe_idx)
            safe_rec_samples.append(safe_rec)
        else:
            safe_rec_samples.append(0.0)
            
    acc_samples = np.array(acc_samples) * 100.0
    safe_rec_samples = np.array(safe_rec_samples) * 100.0
    
    acc_ci = (float(np.percentile(acc_samples, 2.5)), float(np.percentile(acc_samples, 97.5)))
    safe_rec_ci = (float(np.percentile(safe_rec_samples, 2.5)), float(np.percentile(safe_rec_samples, 97.5)))
    
    return {
        "accuracy_mean": float(np.mean(acc_samples)),
        "accuracy_ci_95": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
        "safe_recall_mean": float(np.mean(safe_rec_samples)),
        "safe_recall_ci_95": [round(safe_rec_ci[0], 4), round(safe_rec_ci[1], 4)]
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dedup_path = os.path.join(ROOT_DIR, "data", "processed", "combined_test_dedup.parquet")
    df = pd.read_parquet(dedup_path)
    texts = df["cleaned_text"].tolist()
    labels = df["cyberbullying_type"].tolist()
    y_true = np.array([LABEL_MAP[l] for l in labels])
    n_samples = len(y_true)
    print(f"Loaded {n_samples} samples from {dedup_path}")

    # Compute probabilities for all 7 models
    probs = {}
    for name, path in MODELS.items():
        print(f"Inference on {name}...")
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        probs[name] = predict_probs(mod, tok, texts, device, batch_size=64)
        del tok, mod
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Probability vectors
    p_archive = probs["archive"]
    p_ens_a = (probs["archive"] + probs["run0"] + probs["run0b"]) / 3.0
    p_ens_d = (probs["archive"] + probs["run0"] + probs["run0b"] +
               probs["b_s42"] + probs["b_s43"] + probs["c_s42"] + probs["c_s43"]) / 7.0

    # Predictions (argmax mode)
    pred_archive = np.argmax(p_archive, axis=1)
    pred_a = np.argmax(p_ens_a, axis=1)
    pred_d = np.argmax(p_ens_d, axis=1)

    # Correctness boolean arrays
    corr_archive = (pred_archive == y_true)
    corr_a = (pred_a == y_true)
    corr_d = (pred_d == y_true)

    # Point estimates
    acc_arch = float(np.mean(corr_archive) * 100.0)
    acc_a = float(np.mean(corr_a) * 100.0)
    acc_d = float(np.mean(corr_d) * 100.0)

    safe_mask = (y_true == SAFE_IDX)
    safe_rec_arch = float(np.mean(pred_archive[safe_mask] == SAFE_IDX) * 100.0)
    safe_rec_a = float(np.mean(pred_a[safe_mask] == SAFE_IDX) * 100.0)
    safe_rec_d = float(np.mean(pred_d[safe_mask] == SAFE_IDX) * 100.0)

    print("\n=================== POINT ESTIMATES (ARGMAX MODE) ===================")
    print(f"Archive      : Accuracy = {acc_arch:.4f}%, Safe Recall = {safe_rec_arch:.4f}%")
    print(f"Ensemble (a) : Accuracy = {acc_a:.4f}%, Safe Recall = {safe_rec_a:.4f}%")
    print(f"Ensemble (d) : Accuracy = {acc_d:.4f}%, Safe Recall = {safe_rec_d:.4f}%")

    print("\n=================== 1. EXACT MCNEMAR TESTS ===================")
    # Comparison 1: Ensemble (a) vs Archive
    mcnemar_a_vs_arch = compute_exact_mcnemar(corr_archive, corr_a, name1="Archive", name2="Ensemble (a)")
    print("--- Ensemble (a) vs Archive ---")
    print(f"  Contingency Table: a={mcnemar_a_vs_arch['both_correct_a']}, b={mcnemar_a_vs_arch['m1_correct_m2_incorrect_b']}, c={mcnemar_a_vs_arch['m1_incorrect_m2_correct_c']}, d={mcnemar_a_vs_arch['both_incorrect_d']}")
    print(f"  Archive correct & Ensemble (a) incorrect (b) : {mcnemar_a_vs_arch['m1_correct_m2_incorrect_b']}")
    print(f"  Archive incorrect & Ensemble (a) correct (c) : {mcnemar_a_vs_arch['m1_incorrect_m2_correct_c']}")
    print(f"  Total discordant pairs (b + c)               : {mcnemar_a_vs_arch['discordant_pairs_n']}")
    print(f"  Exact McNemar Two-Sided p-value             : {mcnemar_a_vs_arch['exact_p_value']:.6e}")

    # Comparison 2: Ensemble (d) vs Ensemble (a)
    mcnemar_d_vs_a = compute_exact_mcnemar(corr_a, corr_d, name1="Ensemble (a)", name2="Ensemble (d)")
    print("\n--- Ensemble (d) vs Ensemble (a) ---")
    print(f"  Contingency Table: a={mcnemar_d_vs_a['both_correct_a']}, b={mcnemar_d_vs_a['m1_correct_m2_incorrect_b']}, c={mcnemar_d_vs_a['m1_incorrect_m2_correct_c']}, d={mcnemar_d_vs_a['both_incorrect_d']}")
    print(f"  Ensemble (a) correct & Ensemble (d) incorrect (b) : {mcnemar_d_vs_a['m1_correct_m2_incorrect_b']}")
    print(f"  Ensemble (a) incorrect & Ensemble (d) correct (c) : {mcnemar_d_vs_a['m1_incorrect_m2_correct_c']}")
    print(f"  Total discordant pairs (b + c)                   : {mcnemar_d_vs_a['discordant_pairs_n']}")
    print(f"  Exact McNemar Two-Sided p-value                 : {mcnemar_d_vs_a['exact_p_value']:.6e}")

    print("\n=================== 2. BOOTSTRAP 95% CIs (1,000 RESAMPLES, SEED 42) ===================")
    boot_arch = bootstrap_ci(pred_archive, y_true, SAFE_IDX, n_bootstraps=1000, seed=42)
    boot_a = bootstrap_ci(pred_a, y_true, SAFE_IDX, n_bootstraps=1000, seed=42)
    boot_d = bootstrap_ci(pred_d, y_true, SAFE_IDX, n_bootstraps=1000, seed=42)

    print("--- Archive ---")
    print(f"  Accuracy    : {acc_arch:.4f}% | 95% CI: [{boot_arch['accuracy_ci_95'][0]:.4f}%, {boot_arch['accuracy_ci_95'][1]:.4f}%]")
    print(f"  Safe Recall : {safe_rec_arch:.4f}% | 95% CI: [{boot_arch['safe_recall_ci_95'][0]:.4f}%, {boot_arch['safe_recall_ci_95'][1]:.4f}%]")

    print("\n--- Ensemble (a) ---")
    print(f"  Accuracy    : {acc_a:.4f}% | 95% CI: [{boot_a['accuracy_ci_95'][0]:.4f}%, {boot_a['accuracy_ci_95'][1]:.4f}%]")
    print(f"  Safe Recall : {safe_rec_a:.4f}% | 95% CI: [{boot_a['safe_recall_ci_95'][0]:.4f}%, {boot_a['safe_recall_ci_95'][1]:.4f}%]")

    print("\n--- Ensemble (d) ---")
    print(f"  Accuracy    : {acc_d:.4f}% | 95% CI: [{boot_d['accuracy_ci_95'][0]:.4f}%, {boot_d['accuracy_ci_95'][1]:.4f}%]")
    print(f"  Safe Recall : {safe_rec_d:.4f}% | 95% CI: [{boot_d['safe_recall_ci_95'][0]:.4f}%, {boot_d['safe_recall_ci_95'][1]:.4f}%]")

    # Save to experiments/
    results_payload = {
        "metadata": {
            "dataset": "data/processed/combined_test_dedup.parquet",
            "samples_count": n_samples,
            "mode": "argmax",
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 42
        },
        "point_estimates": {
            "archive": {"accuracy": acc_arch, "safe_recall": safe_rec_arch},
            "ensemble_a": {"accuracy": acc_a, "safe_recall": safe_rec_a},
            "ensemble_d": {"accuracy": acc_d, "safe_recall": safe_rec_d}
        },
        "mcnemar_tests": {
            "ensemble_a_vs_archive": mcnemar_a_vs_arch,
            "ensemble_d_vs_ensemble_a": mcnemar_d_vs_a
        },
        "bootstrap_95_ci": {
            "archive": boot_arch,
            "ensemble_a": boot_a,
            "ensemble_d": boot_d
        }
    }

    out_file = os.path.join(ROOT_DIR, "experiments", "mcnemar_and_bootstrap_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved results successfully to: {out_file}")

if __name__ == "__main__":
    main()

import os
import sys
import gc
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

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
CLASS_NAMES = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]
BULLY_CLASSES = ["age", "ethnicity", "gender", "other_cyberbullying", "religion"]

MODELS = {
    "archive": os.path.join(ROOT_DIR, "models", "muril_cyberbullying_v2_stable_archive_0e904b5"),
    "run0": os.path.join(ROOT_DIR, "models", "muril_exp_run0_control"),
    "run0b": os.path.join(ROOT_DIR, "models", "muril_exp_run0b_seed43"),
    "b_s42": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_B_all"),
    "b_s43": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_B_seed43"),
    "c_s42": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_C_oversample3x"),
    "c_s43": os.path.join(ROOT_DIR, "models", "muril_exp_ablation_C_seed43"),
}

ENSEMBLES = {
    "(a) Archive+Run0+Run0b": ["archive", "run0", "run0b"],
    "(b) B s42+B s43": ["b_s42", "b_s43"],
    "(c) C s42+C s43": ["c_s42", "c_s43"],
    "(d) All 7": ["archive", "run0", "run0b", "b_s42", "b_s43", "c_s42", "c_s43"]
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

def evaluate_predictions(preds, true_labels, is_human=False):
    preds = np.array(preds)
    true_labels = np.array(true_labels)
    acc = float(accuracy_score(true_labels, preds) * 100.0)
    
    report = classification_report(true_labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    safe_rec = float(report["not_cyberbullying"]["recall"] * 100.0)
    other_rec = float(report["other_cyberbullying"]["recall"] * 100.0)
    
    bully_recalls = [report[c]["recall"] * 100.0 for c in BULLY_CLASSES]
    bully_macro_rec = float(np.mean(bully_recalls))
    
    bully_f1s = [report[c]["f1-score"] * 100.0 for c in BULLY_CLASSES]
    bully_macro_f1 = float(np.mean(bully_f1s))
    
    all_f1s = [report[c]["f1-score"] * 100.0 for c in CLASS_NAMES]
    macro_6class_f1 = float(np.mean(all_f1s))
    
    # Binary metrics (Bully vs Safe)
    is_bully_true = (true_labels != "not_cyberbullying")
    is_bully_pred = (preds != "not_cyberbullying")
    human_bully_rec = float(np.sum(is_bully_true & is_bully_pred) / max(np.sum(is_bully_true), 1) * 100.0)
    
    return {
        "accuracy": acc,
        "safe_recall": safe_rec,
        "other_cyberbullying_recall": other_rec,
        "bully_macro_recall": bully_macro_rec,
        "bully_macro_f1": bully_macro_f1,
        "macro_6class_f1": macro_6class_f1,
        "human_bully_recall": human_bully_rec,
        "report": report
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    dedup_path = os.path.join(ROOT_DIR, "data", "processed", "combined_test_dedup.parquet")
    print(f"Loading deduplicated test set from: {dedup_path}")
    dedup_df = pd.read_parquet(dedup_path)
    dedup_texts = dedup_df["cleaned_text"].tolist()
    dedup_labels = dedup_df["cyberbullying_type"].tolist()
    print(f"Dedup test set size: {len(dedup_df):,} samples")

    human_path = os.path.join(ROOT_DIR, "data", "validation", "hardneg_human_test_set.csv")
    print(f"Loading human test set from: {human_path}")
    human_df = pd.read_csv(human_path)
    human_texts = human_df["text"].tolist()
    human_labels = human_df["category"].tolist()
    print(f"Human test set size: {len(human_df):,} samples")

    # Storage for model probabilities
    dedup_model_probs = {}
    human_model_probs = {}

    for name, path in MODELS.items():
        print(f"\n--- Running inference for model: {name} ({os.path.basename(path)}) ---")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device)

        d_probs = predict_probs(model, tokenizer, dedup_texts, device, batch_size=64)
        h_probs = predict_probs(model, tokenizer, human_texts, device, batch_size=64)

        dedup_model_probs[name] = d_probs
        human_model_probs[name] = h_probs

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=================== ENSEMBLE EVALUATIONS (ARGMAX MODE) ===================")
    results = {}

    for ens_name, model_keys in ENSEMBLES.items():
        print(f"\nEvaluating Ensemble: {ens_name} ({', '.join(model_keys)})")
        
        # Probability averaging
        avg_d_probs = np.mean([dedup_model_probs[k] for k in model_keys], axis=0)
        avg_h_probs = np.mean([human_model_probs[k] for k in model_keys], axis=0)

        # Argmax prediction
        d_preds = [ID_TO_LABEL[idx] for idx in np.argmax(avg_d_probs, axis=1)]
        h_preds = [ID_TO_LABEL[idx] for idx in np.argmax(avg_h_probs, axis=1)]

        d_metrics = evaluate_predictions(d_preds, dedup_labels, is_human=False)
        h_metrics = evaluate_predictions(h_preds, human_labels, is_human=True)

        print(f"--- Dedup Blind Test (N={len(dedup_labels)}) ---")
        print(f"  Accuracy                : {d_metrics['accuracy']:.4f}%")
        print(f"  Safe Recall             : {d_metrics['safe_recall']:.4f}%")
        print(f"  6-Class Macro-F1        : {d_metrics['macro_6class_f1']:.4f}%")
        print(f"  Bully 5-Class Macro-F1  : {d_metrics['bully_macro_f1']:.4f}%")
        print(f"  Other Bully Recall      : {d_metrics['other_cyberbullying_recall']:.4f}%")

        print(f"--- 49-Row Human Set (N={len(human_labels)}) ---")
        print(f"  Safe Recall (30 rows)   : {h_metrics['safe_recall']:.4f}%")
        print(f"  Bully Recall (19 rows)  : {h_metrics['human_bully_recall']:.4f}%")
        print(f"  Accuracy                : {h_metrics['accuracy']:.4f}%")
        print(f"  6-Class Macro-F1        : {h_metrics['macro_6class_f1']:.4f}%")

        results[ens_name] = {
            "models_included": model_keys,
            "dedup_blind_test": {
                "accuracy": d_metrics["accuracy"],
                "safe_recall": d_metrics["safe_recall"],
                "macro_6class_f1": d_metrics["macro_6class_f1"],
                "bully_macro_f1": d_metrics["bully_macro_f1"],
                "other_cyberbullying_recall": d_metrics["other_cyberbullying_recall"],
                "per_class_f1": {c: d_metrics["report"][c]["f1-score"] for c in CLASS_NAMES},
                "per_class_recall": {c: d_metrics["report"][c]["recall"] for c in CLASS_NAMES}
            },
            "human_test_set": {
                "accuracy": h_metrics["accuracy"],
                "safe_recall": h_metrics["safe_recall"],
                "bully_recall": h_metrics["human_bully_recall"],
                "macro_6class_f1": h_metrics["macro_6class_f1"],
                "other_cyberbullying_recall": h_metrics["other_cyberbullying_recall"]
            }
        }

    # Save to experiments/
    out_json = os.path.join(ROOT_DIR, "experiments", "ensemble_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ensemble evaluation results to: {out_json}")

if __name__ == "__main__":
    main()

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import classify_probabilities

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
SAFE_LABEL = "not_cyberbullying"
OTHER_LABEL = "other_cyberbullying"

ARCHIVE_PATH = os.path.join(ROOT_DIR, "models", "muril_cyberbullying_v2_stable_archive_0e904b5")

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

def evaluate_predictions(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    rep = classification_report(y_true, y_pred, labels=CLASS_NAMES, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    macro_prec = float(rep["macro avg"]["precision"] * 100.0)
    macro_rec = float(rep["macro avg"]["recall"] * 100.0)
    macro_f1 = float(rep["macro avg"]["f1-score"] * 100.0)
    
    safe_rec = float(rep[SAFE_LABEL]["recall"] * 100.0)
    safe_prec = float(rep[SAFE_LABEL]["precision"] * 100.0)
    safe_f1 = float(rep[SAFE_LABEL]["f1-score"] * 100.0)
    safe_correct = int(np.sum((np.array(y_true) == SAFE_LABEL) & (np.array(y_pred) == SAFE_LABEL)))
    total_safe = int(np.sum(np.array(y_true) == SAFE_LABEL))
    
    other_rec = float(rep[OTHER_LABEL]["recall"] * 100.0)
    other_prec = float(rep[OTHER_LABEL]["precision"] * 100.0)
    other_f1 = float(rep[OTHER_LABEL]["f1-score"] * 100.0)
    
    bully_f1s = [rep[c]["f1-score"] * 100.0 for c in BULLY_CLASSES]
    bully_macro_f1 = float(np.mean(bully_f1s))
    
    total_correct = int(np.sum(np.array(y_true) == np.array(y_pred)))
    total_samples = len(y_true)

    return {
        "accuracy": acc,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "safe_recall": safe_rec,
        "safe_precision": safe_prec,
        "safe_f1": safe_f1,
        "safe_correct": safe_correct,
        "total_safe": total_safe,
        "other_cyberbullying_recall": other_rec,
        "other_cyberbullying_precision": other_prec,
        "other_cyberbullying_f1": other_f1,
        "bully_macro_f1": bully_macro_f1,
        "per_class_f1": {c: float(rep[c]["f1-score"] * 100.0) for c in CLASS_NAMES},
        "per_class_recall": {c: float(rep[c]["recall"] * 100.0) for c in CLASS_NAMES},
        "per_class_precision": {c: float(rep[c]["precision"] * 100.0) for c in CLASS_NAMES}
    }

def run_eval_on_dataset(df_path, model, tokenizer, device):
    df = pd.read_parquet(df_path)
    n_samples = len(df)
    print(f"\n========================================================")
    print(f"Dataset: {os.path.basename(df_path)} (N = {n_samples})")
    print(f"========================================================")
    
    raw_texts = df["tweet_text"].tolist() if "tweet_text" in df.columns else df["cleaned_text"].tolist()
    cleaned_texts = df["cleaned_text"].tolist()
    labels = df["cyberbullying_type"].tolist()

    probs = predict_probs(model, tokenizer, cleaned_texts, device, batch_size=64)

    # (a) Argmax
    preds_argmax = [ID_TO_LABEL[i] for i in np.argmax(probs, axis=1)]
    res_argmax = evaluate_predictions(labels, preds_argmax)

    # (b) Deployed Two-Stage (safe_threshold=0.50, use_safety_net=True, raw_text passed)
    dec_two_stage = [
        classify_probabilities(p, ID_TO_LABEL, method="two_stage", safe_threshold=0.50, raw_text=rt, use_safety_net=True)
        for p, rt in zip(probs, raw_texts)
    ]
    preds_two_stage = [d["pred_class"] for d in dec_two_stage]
    res_two_stage = evaluate_predictions(labels, preds_two_stage)

    print(f"\n--- (a) ARGMAX MODE ---")
    print(f"  Accuracy                : {res_argmax['accuracy']:.4f}% ({res_argmax['total_correct']}/{res_argmax['total_samples']})")
    print(f"  Macro Precision (6-Cl)  : {res_argmax['macro_precision']:.4f}%")
    print(f"  Macro Recall (6-Cl)     : {res_argmax['macro_recall']:.4f}%")
    print(f"  Macro F1 (6-Cl)         : {res_argmax['macro_f1']:.4f}%")
    print(f"  Safe Recall             : {res_argmax['safe_recall']:.4f}% ({res_argmax['safe_correct']}/{res_argmax['total_safe']})")
    print(f"  Safe Precision          : {res_argmax['safe_precision']:.4f}%")
    print(f"  Other Bully Recall      : {res_argmax['other_cyberbullying_recall']:.4f}%")
    print(f"  Bully Macro F1 (5-Cl)   : {res_argmax['bully_macro_f1']:.4f}%")

    print(f"\n--- (b) DEPLOYED TWO-STAGE MODE (thr=0.50, safety-net=True) ---")
    print(f"  Accuracy                : {res_two_stage['accuracy']:.4f}% ({res_two_stage['total_correct']}/{res_two_stage['total_samples']})")
    print(f"  Macro Precision (6-Cl)  : {res_two_stage['macro_precision']:.4f}%")
    print(f"  Macro Recall (6-Cl)     : {res_two_stage['macro_recall']:.4f}%")
    print(f"  Macro F1 (6-Cl)         : {res_two_stage['macro_f1']:.4f}%")
    print(f"  Safe Recall             : {res_two_stage['safe_recall']:.4f}% ({res_two_stage['safe_correct']}/{res_two_stage['total_safe']})")
    print(f"  Safe Precision          : {res_two_stage['safe_precision']:.4f}%")
    print(f"  Other Bully Recall      : {res_two_stage['other_cyberbullying_recall']:.4f}%")
    print(f"  Bully Macro F1 (5-Cl)   : {res_two_stage['bully_macro_f1']:.4f}%")

    return {
        "n_samples": n_samples,
        "argmax": res_argmax,
        "deployed_two_stage": res_two_stage
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading Archive Model from: {ARCHIVE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(ARCHIVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(ARCHIVE_PATH).to(device)

    # 1. Full test set (N = 5,242)
    full_test_path = os.path.join(ROOT_DIR, "data", "processed", "combined_test.parquet")
    results_full = run_eval_on_dataset(full_test_path, model, tokenizer, device)

    # 2. Deduplicated test set (N = 4,963)
    dedup_test_path = os.path.join(ROOT_DIR, "data", "processed", "combined_test_dedup.parquet")
    results_dedup = run_eval_on_dataset(dedup_test_path, model, tokenizer, device)

    # Save to experiments/full_test_both_modes.json
    out_payload = {
        "model": "models/muril_cyberbullying_v2_stable_archive_0e904b5",
        "checkpoint_sha256": "0b671e02fbb1b933d10ee979bca402778bb18d454fa9156f2f98642a8b301b0f",
        "full_test_set_n5242": results_full,
        "dedup_test_set_n4963": results_dedup
    }

    out_file = os.path.join(ROOT_DIR, "experiments", "full_test_both_modes.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSuccessfully serialized results to: {out_file}")

if __name__ == "__main__":
    main()

"""
Unified Evaluation Script for MuRIL v2 Cyberbullying Detection System.

Evaluates the held-out test set using:
1. Production Two-Stage Safety Boundary (calls src.model.classify_probabilities)
2. Standard 6-Class Argmax Baseline (for academic paper baseline comparison)

Usage:
    python evaluate.py [--batch_size 32] [--max_samples N]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import classify_probabilities

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MuRIL v2 on Held-Out Test Set")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for quick test")
    parser.add_argument("--model_dir", type=str, default="models/muril_cyberbullying_v2", help="Path to model directory")
    parser.add_argument("--test_data", type=str, default="data/processed/combined_test.csv", help="Path to test CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 80)
    print(" 🛡️ UNIFIED EVALUATION SUITE: MURIL V2 MULTILINGUAL 6-CLASS")
    print("=" * 80)

    # 1. Check & load test data
    test_path = os.path.join(ROOT_DIR, args.test_data)
    if not os.path.exists(test_path):
        parquet_path = test_path.replace(".csv", ".parquet")
        if os.path.exists(parquet_path):
            test_path = parquet_path
        else:
            raise FileNotFoundError(f"Could not find test dataset at {test_path}")

    print(f"Loading test set from: {test_path}")
    if test_path.endswith(".csv"):
        test_df = pd.read_csv(test_path)
    else:
        test_df = pd.read_parquet(test_path)

    if args.max_samples and len(test_df) > args.max_samples:
        print(f"Subsampling to {args.max_samples} samples for quick verification...")
        test_df = test_df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    print(f"Total evaluation samples: {len(test_df):,}")

    # 2. Load Model & Tokenizer
    model_path = os.path.join(ROOT_DIR, args.model_dir)
    if not os.path.exists(model_path):
        model_path = "suyashsahu00/muril-cyberbullying-detection"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_path}")
    print(f"Execution Device : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # 3. Label maps
    label_map_file = os.path.join(model_path, "label_map.json") if os.path.isdir(model_path) else None
    if label_map_file and os.path.exists(label_map_file):
        with open(label_map_file, "r") as f:
            lm_data = json.load(f)
            id_to_label = {int(k): v for k, v in lm_data.get("id_to_label", {}).items()}
            label_to_id = lm_data.get("label_map", {})
    else:
        id_to_label = {int(k): v for k, v in model.config.id2label.items()}
        label_to_id = {v: k for k, v in id_to_label.items()}

    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    ground_truth_labels = [label_to_id[t] for t in test_df["cyberbullying_type"]]

    # 4. Inference in Batches
    print(f"\nRunning model inference (batch_size={args.batch_size})...")
    texts = [str(t) if pd.notna(t) else "" for t in test_df["cleaned_text"].values]
    
    all_probs = []
    t0 = time.time()
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        enc = tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.extend(probs)

    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.2f}s ({len(texts)/elapsed:.1f} samples/sec)")

    # 5. Evaluate Option (a) [Production Two-Stage] & Option (b) [Argmax Baseline]
    preds_two_stage = []
    preds_argmax = []

    for p in all_probs:
        # Call unified decision function from src.model
        dec_ts = classify_probabilities(p, id_to_label, method="two_stage", use_safety_net=False)
        preds_two_stage.append(dec_ts["pred_idx"])

        dec_arg = classify_probabilities(p, id_to_label, method="argmax", use_safety_net=False)
        preds_argmax.append(dec_arg["pred_idx"])

    # Compute Metrics
    acc_ts = accuracy_score(ground_truth_labels, preds_two_stage)
    p_ts, r_ts, f1_ts, _ = precision_recall_fscore_support(ground_truth_labels, preds_two_stage, average='macro', zero_division=0)

    acc_arg = accuracy_score(ground_truth_labels, preds_argmax)
    p_arg, r_arg, f1_arg, _ = precision_recall_fscore_support(ground_truth_labels, preds_argmax, average='macro', zero_division=0)

    # 6. Display Reports
    print("\n" + "=" * 80)
    print(" 🚀 OPTION (A): PRODUCTION TWO-STAGE SAFETY BOUNDARY METRICS")
    print("=" * 80)
    print(f"Overall Accuracy : {acc_ts * 100:.2f}%")
    print(f"Macro Precision  : {p_ts * 100:.2f}%")
    print(f"Macro Recall     : {r_ts * 100:.2f}%")
    print(f"Macro F1-Score   : {f1_ts * 100:.2f}%\n")
    print(classification_report(ground_truth_labels, preds_two_stage, target_names=class_names, digits=4))

    print("\n" + "=" * 80)
    print(" 📊 OPTION (B) COMPARISON: ARGMAX BASELINE VS PRODUCTION SAFETY-BOUNDARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Argmax Baseline':<20} | {'Production Two-Stage':<20} | {'Delta':<10}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<25} | {acc_arg*100:>18.2f}% | {acc_ts*100:>18.2f}% | {(acc_ts - acc_arg)*100:>+8.2f}%")
    print(f"{'Macro Precision':<25} | {p_arg*100:>18.2f}% | {p_ts*100:>18.2f}% | {(p_ts - p_arg)*100:>+8.2f}%")
    print(f"{'Macro Recall':<25} | {r_arg*100:>18.2f}% | {r_ts*100:>18.2f}% | {(r_ts - r_arg)*100:>+8.2f}%")
    print(f"{'Macro F1-Score':<25} | {f1_arg*100:>18.2f}% | {f1_ts*100:>18.2f}% | {(f1_ts - f1_arg)*100:>+8.2f}%")
    print("-" * 80)

    # Save to json
    results_payload = {
        "production_two_stage": {
            "accuracy": round(acc_ts * 100, 2),
            "macro_precision": round(p_ts * 100, 2),
            "macro_recall": round(r_ts * 100, 2),
            "macro_f1": round(f1_ts * 100, 2),
            "report": classification_report(ground_truth_labels, preds_two_stage, target_names=class_names, output_dict=True)
        },
        "argmax_baseline": {
            "accuracy": round(acc_arg * 100, 2),
            "macro_precision": round(p_arg * 100, 2),
            "macro_recall": round(r_arg * 100, 2),
            "macro_f1": round(f1_arg * 100, 2),
            "report": classification_report(ground_truth_labels, preds_argmax, target_names=class_names, output_dict=True)
        }
    }

    out_file = os.path.join(ROOT_DIR, "models", "evaluation_metrics_comparison.json")
    with open(out_file, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved metrics comparison to: {out_file}")

if __name__ == "__main__":
    main()

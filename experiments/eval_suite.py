import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support

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
BULLY_CLASSES = ['age', 'ethnicity', 'gender', 'other_cyberbullying', 'religion']

MUST_PASS_SENTENCES = [
    "The new parking permit system at the complex has reduced unauthorized vehicles from six to zero this month.",
    "Subah se bijli nahi thi toh ghar pe hi kaam kiya fortunately net backup tha laptop pe.",
    "आज बाज़ार में मौसमी आम काफी सस्ते मिल रहे थे दो किलो ले आया हूँ घर के लिए।",
    "She recommended a great documentary about the restoration of old Mughal-era stepwells across Rajasthan.",
    "Bhai kal ke match ke liye fantasy team lock karne se pehle pitch report zaroor dekh lena"
]

EXPLICIT_ABUSE = [
    "madarchod",
    "bhosdiwala",
    "you fucking idiot",
    "isko toh chappal khol ke marna chahiye"
]

def predict_batch_probs(model, tokenizer, texts, device, batch_size=64, max_len=128):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[i:i+batch_size]]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)

def evaluate_dataset(probs, raw_texts, true_labels, mode="production"):
    preds = []
    is_bully_preds = []
    
    for i, p in enumerate(probs):
        raw_t = raw_texts[i] if raw_texts is not None else None
        if mode == "production":
            dec = classify_probabilities(p, ID_TO_LABEL, method="two_stage", safe_threshold=0.50, raw_text=raw_t, use_safety_net=True)
        elif mode == "argmax":
            dec = classify_probabilities(p, ID_TO_LABEL, method="argmax")
        elif mode == "threshold040":
            dec = classify_probabilities(p, ID_TO_LABEL, method="two_stage", safe_threshold=0.40, raw_text=None, use_safety_net=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        preds.append(dec["pred_class"])
        is_bully_preds.append(dec["is_cyberbullying"])
        
    preds = np.array(preds)
    is_bully_preds = np.array(is_bully_preds)
    
    report = classification_report(true_labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    bully_recalls = [report[c]["recall"] * 100.0 for c in BULLY_CLASSES]
    bully_macro_recall = np.mean(bully_recalls)
    bully_f1s = [report[c]["f1-score"] * 100.0 for c in BULLY_CLASSES]
    bully_macro_f1 = np.mean(bully_f1s)
    
    safe_recall = report["not_cyberbullying"]["recall"] * 100.0
    safe_f1 = report["not_cyberbullying"]["f1-score"] * 100.0
    safe_correct = int(np.sum((np.array(true_labels) == "not_cyberbullying") & (preds == "not_cyberbullying")))
    total_safe = int(np.sum(np.array(true_labels) == "not_cyberbullying"))
    
    is_bully_true = (np.array(true_labels) != "not_cyberbullying")
    binary_prec, binary_rec, binary_f1, _ = precision_recall_fscore_support(
        is_bully_true, is_bully_preds, average='binary', zero_division=0
    )
    bully_correct = int(np.sum(is_bully_true & is_bully_preds))
    total_bully = int(np.sum(is_bully_true))
    
    overall_acc = float(np.mean(np.array(true_labels) == preds) * 100.0)
    
    return {
        "overall_accuracy": overall_acc,
        "report": report,
        "bully_macro_recall": bully_macro_recall,
        "bully_macro_f1": bully_macro_f1,
        "safe_recall": safe_recall,
        "safe_f1": safe_f1,
        "safe_correct": safe_correct,
        "total_safe": total_safe,
        "bully_correct": bully_correct,
        "total_bully": total_bully,
        "binary_recall": binary_rec * 100.0,
        "binary_precision": binary_prec * 100.0,
        "binary_f1": binary_f1 * 100.0,
        "preds": preds
    }

def run_evaluation_suite(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_dir} on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    
    # 1. Blind Test Set (5,242 rows)
    test_df = pd.read_parquet("data/processed/combined_test.parquet")
    test_texts = test_df["cleaned_text"].tolist()
    test_labels = test_df["cyberbullying_type"].tolist()
    print("Computing probabilities on Full Blind Test Set (5,242 samples)...")
    test_probs = predict_batch_probs(model, tokenizer, test_texts, device)
    
    # 2. Deduplicated Blind Test Set (4,963 rows)
    dedup_df = pd.read_parquet("data/processed/combined_test_dedup.parquet")
    dedup_texts = dedup_df["cleaned_text"].tolist()
    dedup_labels = dedup_df["cyberbullying_type"].tolist()
    print("Computing probabilities on Deduplicated Blind Test Set (4,963 samples)...")
    dedup_probs = predict_batch_probs(model, tokenizer, dedup_texts, device)
    
    # 3. Human Test Set (49 rows)
    human_df = pd.read_csv("data/validation/hardneg_human_test_set.csv")
    human_texts = human_df["text"].tolist()
    human_labels = human_df["category"].tolist()
    print("Computing probabilities on Human Test Set (49 samples)...")
    human_probs = predict_batch_probs(model, tokenizer, human_texts, device)
    
    # 4. Synth Test Set (500 rows)
    synth_df = pd.read_csv("data/validation/hardneg_synth_test_set.csv")
    synth_texts = synth_df["text"].tolist()
    synth_labels = synth_df["category"].tolist()
    print("Computing probabilities on Synth Test Set (500 samples)...")
    synth_probs = predict_batch_probs(model, tokenizer, synth_texts, device)
    
    # 5. Five Must-Pass Sentences
    must_pass_probs = predict_batch_probs(model, tokenizer, MUST_PASS_SENTENCES, device)
    
    # 6. Explicit Abuse
    abuse_probs = predict_batch_probs(model, tokenizer, EXPLICIT_ABUSE, device)
    
    results = {}
    for mode in ["production", "argmax", "threshold040"]:
        res_test = evaluate_dataset(test_probs, test_texts, test_labels, mode=mode)
        res_dedup = evaluate_dataset(dedup_probs, dedup_texts, dedup_labels, mode=mode)
        res_human = evaluate_dataset(human_probs, human_texts, human_labels, mode=mode)
        res_synth = evaluate_dataset(synth_probs, synth_texts, synth_labels, mode=mode)
        
        mp_preds = []
        for i, p in enumerate(must_pass_probs):
            dec = classify_probabilities(p, ID_TO_LABEL, method="two_stage" if mode!="argmax" else "argmax",
                                         safe_threshold=0.50 if mode=="production" else (0.40 if mode=="threshold040" else 0.50),
                                         raw_text=MUST_PASS_SENTENCES[i] if mode=="production" else None,
                                         use_safety_net=(mode=="production"))
            mp_preds.append((MUST_PASS_SENTENCES[i], dec["pred_class"], dec["is_cyberbullying"]))
            
        ab_preds = []
        for i, p in enumerate(abuse_probs):
            dec = classify_probabilities(p, ID_TO_LABEL, method="two_stage" if mode!="argmax" else "argmax",
                                         safe_threshold=0.50 if mode=="production" else (0.40 if mode=="threshold040" else 0.50),
                                         raw_text=EXPLICIT_ABUSE[i] if mode=="production" else None,
                                         use_safety_net=(mode=="production"))
            ab_preds.append((EXPLICIT_ABUSE[i], dec["pred_class"], dec["is_cyberbullying"]))
            
        results[mode] = {
            "test": res_test,
            "dedup_test": res_dedup,
            "human": res_human,
            "synth": res_synth,
            "must_pass": mp_preds,
            "abuse": ab_preds
        }
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()
    
    res = run_evaluation_suite(args.model_dir)
    
    if args.save_json:
        clean_res = {}
        for m in res:
            clean_res[m] = {
                "test_accuracy": res[m]["test"]["overall_accuracy"],
                "test_bully_macro_recall": res[m]["test"]["bully_macro_recall"],
                "test_bully_macro_f1": res[m]["test"]["bully_macro_f1"],
                "test_safe_recall": res[m]["test"]["safe_recall"],
                "test_safe_correct": res[m]["test"]["safe_correct"],
                "test_total_safe": res[m]["test"]["total_safe"],
                "test_f1": {c: res[m]["test"]["report"][c]["f1-score"] for c in CLASS_NAMES},
                "test_recall": {c: res[m]["test"]["report"][c]["recall"] for c in CLASS_NAMES},
                
                "dedup_test_accuracy": res[m]["dedup_test"]["overall_accuracy"],
                "dedup_test_bully_macro_recall": res[m]["dedup_test"]["bully_macro_recall"],
                "dedup_test_bully_macro_f1": res[m]["dedup_test"]["bully_macro_f1"],
                "dedup_test_safe_recall": res[m]["dedup_test"]["safe_recall"],
                "dedup_test_safe_correct": res[m]["dedup_test"]["safe_correct"],
                "dedup_test_total_safe": res[m]["dedup_test"]["total_safe"],
                "dedup_test_f1": {c: res[m]["dedup_test"]["report"][c]["f1-score"] for c in CLASS_NAMES},
                "dedup_test_recall": {c: res[m]["dedup_test"]["report"][c]["recall"] for c in CLASS_NAMES},
                
                "human_safe_recall": res[m]["human"]["safe_recall"],
                "human_safe_correct": res[m]["human"]["safe_correct"],
                "human_total_safe": res[m]["human"]["total_safe"],
                "human_bully_recall": res[m]["human"]["binary_recall"],
                "human_bully_correct": res[m]["human"]["bully_correct"],
                "human_total_bully": res[m]["human"]["total_bully"],
                
                "synth_safe_recall": res[m]["synth"]["safe_recall"],
                "synth_safe_correct": res[m]["synth"]["safe_correct"],
                "synth_total_safe": res[m]["synth"]["total_safe"],
                "synth_bully_recall": res[m]["synth"]["binary_recall"],
                "synth_bully_correct": res[m]["synth"]["bully_correct"],
                "synth_total_bully": res[m]["synth"]["total_bully"],
                
                "must_pass": res[m]["must_pass"],
                "must_pass_safe_count": sum(1 for _, _, is_b in res[m]["must_pass"] if not is_b),
                "abuse": res[m]["abuse"],
                "abuse_detected_count": sum(1 for _, _, is_b in res[m]["abuse"] if is_b)
            }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(clean_res, f, indent=2, ensure_ascii=False)
        print(f"\nSerialized evaluation summary to: {args.save_json}")

import os
import sys
import json
import random
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import get_classifier

def main():
    print("=" * 80)
    # Ensure stdout is UTF-8 for console emojis/special characters
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stdin.encoding != 'utf-8':
        sys.stdin.reconfigure(encoding='utf-8')
        
    print("🧠 CYBERBULLYING DETECTION BLIND TEST TOOL")
    print("=" * 80)
    print("Loading test data and model (this will take a few seconds)...")
    
    # 1. Load test data (using CSV which is faster and has no engine dependencies)
    test_csv = os.path.join(ROOT_DIR, "data", "processed", "combined_test.csv")
    if not os.path.exists(test_csv):
        print(f"Error: Test CSV not found at {test_csv}")
        return
        
    df = pd.read_csv(test_csv)
    
    # Filter for 'other_cyberbullying' samples to quickly find misclassified cases
    other_df = df[df["cyberbullying_type"] == "other_cyberbullying"].copy()
    
    # 2. Get classifier and run predictions on 'other_cyberbullying' samples
    classifier = get_classifier()
    
    print("Running model predictions on 'other_cyberbullying' test set (900 samples)...")
    texts = other_df["cleaned_text"].values
    raw_texts = other_df["text"].values if "text" in other_df.columns else texts
    
    # Run predictions (this is fast because it's only 900 samples)
    preds = []
    for t in texts:
        # We call the internal predict_muril or predict directly
        res = classifier.predict(t, model_choice="muril")
        preds.append(res["category"])  # "Other", "N/A" (meaning Safe), etc.
        
    other_df["pred_category"] = preds
    
    # Find failures where actual is "other_cyberbullying" but predicted is safe ("N/A")
    failures = other_df[other_df["pred_category"] == "N/A"].copy()
    
    if len(failures) == 0:
        print("No misclassified 'other_cyberbullying' examples found. The model is 100% accurate on this subset!")
        return
        
    print(f"Found {len(failures)} actual 'other_cyberbullying' samples misclassified as 'not_cyberbullying' (Safe).")
    print("Randomly selecting 20 samples for your blind test...")
    
    # Randomly sample 20 examples (using fixed seed to be reproducible if needed, but random is better for blind test)
    sample_df = failures.sample(n=min(20, len(failures)), random_state=random.randint(1, 1000))
    
    user_votes = []
    samples_list = []
    
    print("\n" + "=" * 80)
    print("INSTRUCTIONS:")
    print("Read each comment below. Decide if it constitutes cyberbullying or not.")
    print("Type 'b' for Bullying or 'n' for Not Bullying (Safe), then press Enter.")
    print("=" * 80 + "\n")
    
    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        text_content = row.get("text", row["cleaned_text"])
        print(f"[{idx}/20] Comment:")
        print(f"👉 \"{text_content}\"")
        
        while True:
            choice = input("Your Judgment [b/n]: ").strip().lower()
            if choice in ['b', 'n']:
                break
            print("Invalid input. Please enter 'b' (bullying) or 'n' (not bullying).")
            
        user_judgment = "Bullying" if choice == 'b' else "Not Bullying"
        user_votes.append(user_judgment)
        samples_list.append({
            "text": text_content,
            "ground_truth": "other_cyberbullying",  # Since we selected from actual other_cyberbullying
            "model_pred": "not_cyberbullying"       # Since we selected from predicted safe
        })
        print("-" * 50)
        
    # 3. Reveal and compute agreement stats
    print("\n" + "=" * 80)
    print("📊 BLIND TEST RESULTS & COMPARISON")
    print("=" * 80)
    
    agreed_with_gt = 0
    agreed_with_model = 0
    
    for idx, sample in enumerate(samples_list):
        user_j = user_votes[idx]
        gt = "Bullying (Other)"  # Since actual is other_cyberbullying
        model_p = "Not Bullying (Safe)"  # Since predicted is not_cyberbullying
        
        is_user_gt_match = (user_j == "Bullying")
        is_user_model_match = (user_j == "Not Bullying")
        
        if is_user_gt_match:
            agreed_with_gt += 1
        if is_user_model_match:
            agreed_with_model += 1
            
        match_str = ""
        if is_user_gt_match:
            match_str = "✅ You agreed with Ground Truth (Dataset)"
        else:
            match_str = "🤖 You agreed with Model's prediction"
            
        print(f"\n[{idx+1}] Text: \"{sample['text']}\"")
        print(f"    Your Vote   : {user_j}")
        print(f"    Ground Truth: {gt}")
        print(f"    Model Pred  : {model_p}")
        print(f"    Comparison  : {match_str}")
        
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total questions evaluated      : 20")
    print(f"Agreement with Ground Truth    : {agreed_with_gt}/20 ({agreed_with_gt/20*100:.1f}%)")
    print(f"Agreement with Model (Safe)    : {agreed_with_model}/20 ({agreed_with_model/20*100:.1f}%)")
    print("\nInterpretation:")
    if agreed_with_model > agreed_with_gt:
        print("💡 You agreed more with the model than the dataset! This suggests that many of these 'failures'")
        print("   are actually noisy/incorrect labels in the dataset, and the model was correct to flag them safe.")
    else:
        print("💡 You agreed more with the dataset than the model! This suggests these are true model weaknesses")
        print("   where the model missed actual toxic content.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

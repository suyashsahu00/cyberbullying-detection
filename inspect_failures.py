import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Force stdout to use UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Load test data from CSV (no parquet engine dependency)
test_df = pd.read_csv("data/processed/combined_test.csv")
other_test = test_df[test_df["cyberbullying_type"] == "other_cyberbullying"].copy()

print(f"Found {len(other_test)} actual 'other_cyberbullying' samples. Loading MuRIL model...")

# 2. Load model & tokenizer
model_dir = "models/muril_cyberbullying_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

# 3. Label maps
with open(os.path.join(model_dir, "label_map.json"), "r") as f:
    label_data = json.load(f)
    id_to_label = {int(k): v for k, v in label_data["id_to_label"].items()}
    label_to_id = label_data["label_map"]

# 4. Predict
texts = other_test["cleaned_text"].values
raw_texts = other_test["text"].values if "text" in other_test.columns else texts
preds, confs = [], []

for i in range(0, len(texts), 32):
    batch = list(texts[i : i + 32])
    # Handle possible NaN values in cleaned_text
    batch = [str(t) if pd.notna(t) else "" for t in batch]
    enc = tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
    for p in probs:
        not_bully_idx = label_to_id["not_cyberbullying"]
        prob_safe = p[not_bully_idx]
        is_bullying = (1.0 - prob_safe >= 0.5)
        
        if is_bullying:
            bully_indices = [idx for idx in range(6) if idx != not_bully_idx]
            top_idx = bully_indices[int(np.argmax([p[idx] for idx in bully_indices]))]
            preds.append(id_to_label[top_idx])
            confs.append((1.0 - prob_safe) * 100)
        else:
            preds.append("not_cyberbullying")
            confs.append(prob_safe * 100)

other_test["pred"] = preds
other_test["conf"] = confs

# 5. Filter for misclassifications to 'not_cyberbullying'
failures = other_test[other_test["pred"] == "not_cyberbullying"]
print(f"Total misclassified as not_cyberbullying: {len(failures)}")

# Select 50 random samples using pandas sample to avoid ordering/cherry-picking bias
sample_size = min(50, len(failures))
failures_random = failures.sample(n=sample_size, random_state=42)

# Print 50 random examples
print("\n" + "="*80 + f"\n50 RANDOM MISCLASSIFIED EXAMPLES (random_state=42)\n" + "="*80)
for idx, (_, row) in enumerate(failures_random.iterrows(), 1):
    txt = row.get('text', row['cleaned_text'])
    print(f"\n[{idx}] Text: {txt}")
    print(f"    Cleaned: {row['cleaned_text']}")
    print(f"    Confidence: {row['conf']:.2f}%")

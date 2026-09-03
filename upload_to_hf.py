import os
import sys
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

REPO_ID = "suyashsahu00/muril-cyberbullying-detection"
MODEL_DIR = os.path.abspath("models/muril_cyberbullying_v2")

MODEL_CARD_CONTENT = """---
language:
  - en
  - hi
license: mit
tags:
  - cyberbullying-detection
  - text-classification
  - muril
  - bert
  - multilingual
  - explainability
pipeline_tag: text-classification
widget:
  - text: "You are the sweetest and most caring person, thank you!"
    example_title: "Friendly (Non-bullying)"
  - text: "You are too old to be working here, go retire already grandpa."
    example_title: "Age Bullying"
  - text: "Go back to where you came from, your kind isn't welcome here."
    example_title: "Ethnicity Bullying"
  - text: "Women belong in the kitchen and shouldn't speak up."
    example_title: "Gender Bullying"
  - text: "All people following that fake religion are terrorists."
    example_title: "Religion Bullying"
---

# MuRIL Multilingual Cyberbullying Detection (v2)

This model is a fine-tuned version of Google's **MuRIL (Multilingual Representations for Indian Languages)** BERT architecture, specifically adapted for multi-class **Cyberbullying & Hate Speech Detection**.

It categorizes social media text and online commentary across **6 categories**:
1. `age`: Cyberbullying targeting an individual's age.
2. `ethnicity`: Bullying or hate speech targeting ethnicity, race, or caste.
3. `gender`: Misogyny, sexism, or gender-based harassment.
4. `religion`: Hate speech or insults targeting religious beliefs.
5. `other_cyberbullying`: General toxic harassment, insults, or threats.
6. `not_cyberbullying`: Benign, safe, neutral, or positive text.

---

## Model Performance

Evaluated on the held-out multi-class test benchmark:

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **81.97%** |
| **Macro Precision** | **83.21%** |
| **Macro Recall** | **83.41%** |
| **Macro F1-Score** | **83.29%** |

### Per-Class F1 Breakdown:
- **Age**: 97.76% F1
- **Ethnicity**: 95.86% F1
- **Religion**: 95.03% F1
- **Gender**: 86.32% F1
- **Not Cyberbullying**: 64.63% F1
- **Other Cyberbullying**: 60.11% F1

---

## Quickstart & Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_id = "suyashsahu00/muril-cyberbullying-detection"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)

text = "Your message or tweet goes here"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

# Run prediction
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_idx = torch.argmax(probs).item()

predicted_label = model.config.id2label[predicted_idx]
confidence = probs[predicted_idx].item()

print(f"Prediction: {predicted_label} ({confidence * 100:.2f}%)")
```

---

## Architectural Details
- **Base Model:** `google/muril-base-cased`
- **Output Classes:** 6 classes with calibrated `id2label` mappings
- **Weights Format:** SafeTensors (`model.safetensors`)
- **Max Sequence Length:** 128 tokens
"""

def main():
    api = HfApi()
    
    print(f"1. Checking/Creating repository: {REPO_ID}...")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(" Repository ready on Hugging Face!")

    print(f"\n2. Uploading model files from {MODEL_DIR}...")
    # Upload folder excluding local test evaluation jsons
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=["test_evaluation_metrics*.json"]
    )
    print(" Model weights and tokenizer uploaded successfully!")

    print("\n3. Creating and uploading Model Card (README.md)...")
    readme_path = os.path.join(MODEL_DIR, "TEMP_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_CONTENT)
    
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model"
    )
    if os.path.exists(readme_path):
        os.remove(readme_path)
        
    print(" README / Model Card updated!")
    print(f"\n All done! View your model at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()

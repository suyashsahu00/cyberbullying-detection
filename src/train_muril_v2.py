import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Label mappings matching 6-class baseline
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

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc, prec, rec, f1, np.array(all_preds), np.array(all_labels)

def train(epochs=2, batch_size=32, lr=2e-5, max_len=128):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(root_dir, "data", "processed")
    base_model_dir = os.path.join(root_dir, "models", "muril_base_safetensors")
    output_dir = os.path.join(root_dir, "models", "muril_cyberbullying_v2")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" TRAINING GOOGLE MURIL V2 (6-CLASS MULTILINGUAL)")
    print(f" Execution Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Output Directory : {output_dir}")
    print(f" Hyperparameters  : Epochs={epochs}, BatchSize={batch_size}, LR={lr}, MaxLen={max_len}")
    print("=" * 75)

    # 1. Load Data
    train_df = pd.read_parquet(os.path.join(processed_dir, "combined_train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "combined_val.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "combined_test.parquet"))

    print(f"\nLoaded Datasets:")
    print(f"  - Train : {len(train_df):,} samples")
    print(f"  - Val   : {len(val_df):,} samples")
    print(f"  - Test  : {len(test_df):,} samples")

    # 2. Tokenizer & Model
    print(f"\nLoading base tokenizer & model from {base_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP
    )
    model.to(device)

    # Datasets & Loaders
    train_dataset = TextDataset(
        train_df['cleaned_text'].values,
        [LABEL_MAP[t] for t in train_df['cyberbullying_type']],
        tokenizer,
        max_len=max_len
    )
    val_dataset = TextDataset(
        val_df['cleaned_text'].values,
        [LABEL_MAP[t] for t in val_df['cyberbullying_type']],
        tokenizer,
        max_len=max_len
    )
    test_dataset = TextDataset(
        test_df['cleaned_text'].values,
        [LABEL_MAP[t] for t in test_df['cyberbullying_type']],
        tokenizer,
        max_len=max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        print(f"\n>>> Starting Epoch {epoch + 1}/{epochs} ({len(train_loader)} batches) <<<")

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - start_time
                print(f"  Step [{step+1:4d}/{len(train_loader):4d}] | Loss: {loss.item():.4f} | Elapsed: {elapsed/60:.1f}m")

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"--- Epoch {epoch + 1} Summary ---")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Val Acc    : {val_acc * 100:.2f}%")
        print(f"  Val F1     : {val_f1 * 100:.2f}% (Macro)")

        # Save checkpoint if best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"  --> New best validation F1! Saving model checkpoint...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save label map metadata
            with open(os.path.join(output_dir, "label_map.json"), "w") as f:
                json.dump({"label_map": LABEL_MAP, "id_to_label": ID_TO_LABEL}, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n Training finished in {total_time / 60:.2f} minutes!")

    # 4. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 75)
    print(" FINAL HELD-OUT TEST SET EVALUATION REPORT (MuRIL v2 6-Class)")
    print("=" * 75)

    # Load best model for evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(
        best_model, test_loader, criterion, device
    )

    report_str = classification_report(
        test_labels,
        test_preds,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(f"\nOverall Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Macro Precision       : {test_prec * 100:.2f}%")
    print(f"Macro Recall          : {test_rec * 100:.2f}%")
    print(f"Macro F1-Score        : {test_f1 * 100:.2f}%")
    print("\nDetailed Per-Class Classification Report:")
    print(report_str)

    # Save metrics to json
    report_dict = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(f"{'':20s}" + "".join([f"{c[:10]:>12s}" for c in CLASS_NAMES]))
    for idx, row in enumerate(cm):
        print(f"{CLASS_NAMES[idx]:20s}" + "".join([f"{val:12d}" for val in row]))

    results_summary = {
        "model_name": "Google MuRIL v2 (6-Class Multilingual)",
        "overall_accuracy": round(test_acc * 100, 2),
        "macro_precision": round(test_prec * 100, 2),
        "macro_recall": round(test_rec * 100, 2),
        "macro_f1": round(test_f1 * 100, 2),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "training_time_minutes": round(total_time / 60, 2)
    }

    with open(os.path.join(output_dir, "test_evaluation_metrics.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n Evaluation metrics serialized to: {os.path.join(output_dir, 'test_evaluation_metrics.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

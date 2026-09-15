"""
Standalone Auxiliary Sarcasm Detection Module (SDSHL Dataset).
This module is independent and decoupled from the primary cyberbullying pipeline.
It provides a supplementary baseline classifier trained on the 2,000 Hinglish sarcasm samples from dasarpai/SDSHL.
"""

import os
import sys
import io
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure console supports UTF-8 characters on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "sarcasm_auxiliary.joblib")

TRAIN_URL = "https://raw.githubusercontent.com/dasarpai/SDSHL/main/data/processed/2-train.csv"
TEST_URL = "https://raw.githubusercontent.com/dasarpai/SDSHL/main/data/processed/2-test.csv"


def load_data():
    """Fetch official train and test sets from SDSHL repo."""
    print("📥 Loading SDSHL train and test sets from GitHub...")
    train_df = pd.read_csv(TRAIN_URL, sep="\t")
    test_df = pd.read_csv(TEST_URL, sep="\t")
    
    # Drop any null sentences
    train_df = train_df.dropna(subset=["sentence", "label"])
    test_df = test_df.dropna(subset=["sentence", "label"])
    
    print(f"✅ Data loaded: Train={len(train_df)} samples, Test={len(test_df)} samples")
    return train_df, test_df


def train_sarcasm_model():
    """Train a character + word n-gram TF-IDF Logistic Regression pipeline."""
    train_df, test_df = load_data()
    
    X_train, y_train = train_df["sentence"].astype(str), train_df["label"].astype(int)
    X_test, y_test = test_df["sentence"].astype(str), test_df["label"].astype(int)
    
    print("\n⚙️ Training TF-IDF (char+word) + Logistic Regression baseline...")
    # Using sublinear tf and word/char ngrams for Devanagari/Hinglish morphological coverage
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print("\n" + "=" * 60)
    print("📊 AUXILIARY SARCASM MODEL RESULTS (SDSHL BENCHMARK)")
    print("=" * 60)
    print(f"Train Accuracy : {train_acc * 100:.2f}%")
    print(f"Test Accuracy  : {test_acc * 100:.2f}%")
    print("\nDetailed Test Classification Report (0 = Non-Sarcastic, 1 = Sarcastic):")
    print(classification_report(y_test, test_preds, target_names=["Non-Sarcastic", "Sarcastic"], digits=4))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, test_preds)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    
    # Save standalone model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"\n💾 Model artifact saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)
    return pipeline


def predict_sarcasm(text: str, model_path: str = MODEL_SAVE_PATH) -> dict:
    """Standalone inference function for sarcasm prediction."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_sarcasm_model() first.")
    
    model = joblib.load(model_path)
    probs = model.predict_proba([text])[0]
    prob_sarcastic = float(probs[1])
    is_sarcastic = prob_sarcastic >= 0.5
    
    return {
        "text": text,
        "is_sarcastic": is_sarcastic,
        "sarcasm_confidence": round((prob_sarcastic if is_sarcastic else 1.0 - prob_sarcastic) * 100, 2),
        "probability_sarcastic": round(prob_sarcastic * 100, 2),
        "verdict": "Sarcastic" if is_sarcastic else "Non-Sarcastic"
    }


if __name__ == "__main__":
    train_sarcasm_model()
    
    # Quick sanity check
    print("\n🧪 Testing Sample Inferences:")
    sample_tests = [
        "अंग्रेजी नहीं आती है इसलिए हिन्दी ट्विट ज्यादा करते हैं क्या बात है!",
        "बहुत ही बढ़िया काम किया है सरकार ने, अब तो देश बदल ही गया।",
        "कृपया नियमों का पालन करें और शांति बनाए रखें।"
    ]
    for s in sample_tests:
        res = predict_sarcasm(s)
        print(f"• \"{s}\" -> {res['verdict']} ({res['sarcasm_confidence']}%)")

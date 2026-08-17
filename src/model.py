import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import clean_text, detect_language, get_text_stats
from src.explainability import extract_trigger_words, escape_html, TRIGGER_LEXICON

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "baseline_model.pkl")
MURIL_MODEL_DIR = os.path.join(ROOT_DIR, "models", "muril_cyberbullying")

class CyberbullyingSystem:
    """
    Unified Cyberbullying Detection & Explainability Engine.
    Supports both:
    1. Classical Multi-Class Baseline (TF-IDF + Linear SVM) for 6 categories.
    2. Deep Google MuRIL Transformer for Multilingual & Hinglish contextual detection.
    """
    def __init__(self):
        self.baseline_pipeline = None
        self.muril_model = None
        self.muril_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_baseline()
        self._load_muril()

    def _load_baseline(self):
        """Load trained baseline pipeline."""
        if os.path.exists(BASELINE_MODEL_PATH):
            try:
                with open(BASELINE_MODEL_PATH, "rb") as f:
                    self.baseline_pipeline = pickle.load(f)
                print(f" Loaded Baseline Model ({self.baseline_pipeline.classes_})")
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}")

    def _load_muril(self):
        """Load fine-tuned MuRIL transformer model."""
        if os.path.exists(MURIL_MODEL_DIR):
            try:
                self.muril_tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL_DIR)
                self.muril_model = AutoModelForSequenceClassification.from_pretrained(MURIL_MODEL_DIR)
                self.muril_model.to(self.device)
                self.muril_model.eval()
                print(f" Loaded Google MuRIL Model on {self.device}")
            except Exception as e:
                print(f"Warning: Could not load MuRIL model: {e}")

    def predict_baseline(self, raw_text: str) -> Dict[str, Any]:
        """Predict using 6-class Linear Baseline."""
        cleaned = clean_text(raw_text)
        probs = self.baseline_pipeline.predict_proba([cleaned])[0]
        classes = self.baseline_pipeline.classes_
        top_idx = int(np.argmax(probs))
        pred_label = str(classes[top_idx])
        confidence = float(probs[top_idx]) * 100

        is_bullying = (pred_label.lower() != "not_cyberbullying" and pred_label.lower() != "non_bully")
        category_map = {
            "age": "Age",
            "gender": "Gender",
            "ethnicity": "Ethnicity",
            "religion": "Religion",
            "other_cyberbullying": "Other",
            "not_cyberbullying": "N/A"
        }
        display_category = category_map.get(pred_label.lower(), pred_label.capitalize()) if is_bullying else "N/A"
        verdict = "Cyberbullying Detected" if is_bullying else "Not Cyberbullying"

        # Explainability
        explainability = extract_trigger_words(raw_text, display_category if is_bullying else "Other", confidence)
        if not is_bullying:
            explainability["highlighted_text"] = escape_html(raw_text)
            explainability["trigger_words"] = []
            explainability["spans"] = []

        return {
            "verdict": verdict,
            "is_cyberbullying": is_bullying,
            "category": display_category,
            "confidence": round(confidence, 1),
            "model_used": "Classical Baseline (TF-IDF + Linear SVM)",
            "all_probabilities": {str(cls): round(float(p) * 100, 1) for cls, p in zip(classes, probs)},
            "explainability": explainability
        }

    def predict_muril(self, raw_text: str) -> Dict[str, Any]:
        """Predict using fine-tuned Google MuRIL Transformer."""
        cleaned = clean_text(raw_text)
        if self.muril_model is None or self.muril_tokenizer is None:
            return self.predict_baseline(raw_text)

        enc = self.muril_tokenizer(cleaned, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.muril_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        prob_safe = float(probs[0]) * 100
        prob_bully = float(probs[1]) * 100
        is_bullying = (prob_bully >= 50.0)
        confidence = prob_bully if is_bullying else prob_safe
        verdict = "Cyberbullying Detected" if is_bullying else "Not Cyberbullying"

        # Determine category via trigger lexicon or baseline fallback
        category = "Hinglish / Slur" if is_bullying else "N/A"
        for cat, keywords in TRIGGER_LEXICON.items():
            if any(k in cleaned.lower() for k in keywords):
                category = cat
                break

        explainability = extract_trigger_words(raw_text, category if is_bullying else "Other", confidence)
        if not is_bullying:
            explainability["highlighted_text"] = escape_html(raw_text)
            explainability["trigger_words"] = []
            explainability["spans"] = []

        return {
            "verdict": verdict,
            "is_cyberbullying": is_bullying,
            "category": category,
            "confidence": round(confidence, 1),
            "model_used": "Google MuRIL Transformer (Multilingual/Hinglish)",
            "all_probabilities": {
                "Safe / Non-Bully": round(prob_safe, 1),
                "Cyberbullying / Harassment": round(prob_bully, 1)
            },
            "explainability": explainability
        }

    def predict(self, raw_text: str, model_choice: str = "muril") -> Dict[str, Any]:
        """Unified inference entry point with timing and metadata."""
        if not raw_text or not raw_text.strip():
            return {
                "verdict": "Not Cyberbullying",
                "is_cyberbullying": False,
                "category": "N/A",
                "confidence": 100.0,
                "language": "English",
                "original_text": "",
                "cleaned_text": "",
                "latency_ms": 0.1,
                "model_used": "None",
                "all_probabilities": {},
                "explainability": {"spans": [], "highlighted_text": "", "trigger_words": []}
            }

        t0 = time.perf_counter()
        language = detect_language(raw_text)
        stats = get_text_stats(raw_text)

        if model_choice.lower() == "baseline" or self.muril_model is None:
            res = self.predict_baseline(raw_text)
        else:
            res = self.predict_muril(raw_text)

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)

        res.update({
            "language": language,
            "original_text": raw_text,
            "cleaned_text": clean_text(raw_text),
            "latency_ms": latency_ms,
            "stats": stats
        })
        return res

# Global instance
_system_instance = None

def get_classifier() -> CyberbullyingSystem:
    global _system_instance
    if _system_instance is None:
        _system_instance = CyberbullyingSystem()
    return _system_instance

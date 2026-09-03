import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.preprocessing import clean_text, detect_language, get_text_stats
from src.explainability import extract_trigger_words, escape_html, TRIGGER_LEXICON

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "baseline_model.pkl")
MURIL_V2_MODEL_DIR = os.path.join(ROOT_DIR, "models", "muril_cyberbullying_v2")
MURIL_V1_MODEL_DIR = os.path.join(ROOT_DIR, "models", "muril_cyberbullying")

CATEGORY_DISPLAY_MAP = {
    "age": "Age",
    "gender": "Gender",
    "ethnicity": "Ethnicity",
    "religion": "Religion",
    "other_cyberbullying": "Other",
    "not_cyberbullying": "N/A"
}

class CyberbullyingSystem:
    """
    Unified Cyberbullying Detection & Explainability Engine.
    Supports both:
    1. Classical Multi-Class Baseline (TF-IDF + Linear SVM) for 6 categories + Keyword-based trigger detection (~35MB RAM).
    2. Deep Google MuRIL v2 Transformer for 6-Class Multilingual detection + Real Gradient Token Attribution.
    """
    def __init__(self):
        self.baseline_pipeline = None
        self.muril_model = None
        self.muril_tokenizer = None
        self.muril_explainer = None
        self.muril_id2label = None
        self.device = None
        self.is_render = os.environ.get("RENDER", "").lower() in ("true", "1") or os.environ.get("LOW_MEMORY_MODE", "").lower() in ("true", "1")
        
        self._load_baseline()
        if not self.is_render:
            self._load_muril()
        else:
            print(" Render Free Tier detected (< 512MB RAM). Using 6-Class Baseline Pipeline (~35MB RAM).")

    def _load_baseline(self):
        """Load trained baseline pipeline."""
        if os.path.exists(BASELINE_MODEL_PATH):
            try:
                with open(BASELINE_MODEL_PATH, "rb") as f:
                    self.baseline_pipeline = pickle.load(f)
                print(f" Loaded Baseline Model ({getattr(self.baseline_pipeline, 'classes_', [])})")
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}")

    def _load_muril(self):
        """Load fine-tuned MuRIL transformer model (prefers local v2 model, falls back to Hugging Face Hub)."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from src.real_explainability import get_explainer
            
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            HF_REPO_ID = "suyashsahu00/muril-cyberbullying-detection"
            target_path = None
            if os.path.exists(MURIL_V2_MODEL_DIR):
                target_path = MURIL_V2_MODEL_DIR
            elif os.path.exists(MURIL_V1_MODEL_DIR):
                target_path = MURIL_V1_MODEL_DIR
            else:
                target_path = HF_REPO_ID

            print(f" Loading MuRIL model from {target_path}...")
            self.muril_tokenizer = AutoTokenizer.from_pretrained(target_path)
            self.muril_model = AutoModelForSequenceClassification.from_pretrained(target_path)
            self.muril_model.to(self.device)
            self.muril_model.eval()

            # Load label maps if present
            if os.path.isdir(str(target_path)):
                label_map_file = os.path.join(target_path, "label_map.json")
                if os.path.exists(label_map_file):
                    with open(label_map_file, "r") as f:
                        data = json.load(f)
                        self.muril_id2label = {int(k): v for k, v in data.get("id_to_label", {}).items()}
                else:
                    self.muril_id2label = {int(k): v for k, v in self.muril_model.config.id2label.items()}
            else:
                self.muril_id2label = {int(k): v for k, v in self.muril_model.config.id2label.items()}

            # Initialize real gradient-based explainer
            self.muril_explainer = get_explainer(self.muril_model, self.muril_tokenizer)
            print(f" Loaded Google MuRIL Model from {target_path} onto {self.device} (Heads: {self.muril_model.config.num_labels})")
        except Exception as e:
            print(f"Warning: Could not load MuRIL model: {e}")


    def predict_baseline(self, raw_text: str) -> Dict[str, Any]:
        """Predict using 6-class Linear Baseline + Keyword-Based Trigger Detection."""
        cleaned = clean_text(raw_text)
        probs = self.baseline_pipeline.predict_proba([cleaned])[0]
        classes = self.baseline_pipeline.classes_
        top_idx = int(np.argmax(probs))
        pred_label = str(classes[top_idx])
        confidence = float(probs[top_idx]) * 100

        is_bullying = (pred_label.lower() != "not_cyberbullying" and pred_label.lower() != "non_bully")
        display_category = CATEGORY_DISPLAY_MAP.get(pred_label.lower(), pred_label.capitalize()) if is_bullying else "N/A"
        verdict = "Cyberbullying Detected" if is_bullying else "Not Cyberbullying"

        # Keyword-based trigger detection fallback
        explainability = extract_trigger_words(raw_text, display_category if is_bullying else "Other", confidence)
        explainability["method"] = "Keyword-Based Trigger Detection"
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
            "explainability_method": "Keyword-Based Trigger Detection",
            "all_probabilities": {str(cls): round(float(p) * 100, 1) for cls, p in zip(classes, probs)},
            "explainability": explainability
        }

    def predict_muril(self, raw_text: str) -> Dict[str, Any]:
        """Predict using fine-tuned Google MuRIL Transformer + Real Gradient Attribution."""
        cleaned = clean_text(raw_text)
        if self.muril_model is None or self.muril_tokenizer is None:
            return self.predict_baseline(raw_text)

        import torch
        enc = self.muril_tokenizer(cleaned, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.muril_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        num_classes = len(probs)
        safety_net_triggered = False
        safety_net_note = None
        
        # Check if v2 multi-class model (6 classes)
        if num_classes == 6 and self.muril_id2label:
            not_bully_indices = [i for i, name in self.muril_id2label.items() if name.lower() == "not_cyberbullying"]
            not_bully_idx = not_bully_indices[0] if not_bully_indices else 3
            
            prob_safe = float(probs[not_bully_idx]) * 100
            prob_bully = 100.0 - prob_safe
            is_bullying = (prob_bully >= 50.0)

            if is_bullying:
                bully_indices = [i for i in range(num_classes) if i != not_bully_idx]
                top_bully_idx = bully_indices[int(np.argmax([probs[i] for i in bully_indices]))]
                pred_class = str(self.muril_id2label[top_bully_idx])
                confidence = prob_bully
                display_category = CATEGORY_DISPLAY_MAP.get(pred_class.lower(), pred_class.capitalize())
                verdict = "Cyberbullying Detected"
            else:
                # Hybrid Safety-Net fallback for low-margin safe classifications
                if prob_safe < 60.0:
                    from src.explainability import extract_trigger_words, HIGH_SEVERITY_HINGLISH
                    trigger_res = extract_trigger_words(raw_text, "Other", prob_safe)
                    high_severity_matches = [
                        span for span in trigger_res.get("spans", [])
                        if span.get("weight", 0.0) > 0.8 and 
                        (span.get("word", "").lower() in HIGH_SEVERITY_HINGLISH or 
                         any(w in span.get("word", "").lower() for w in HIGH_SEVERITY_HINGLISH))
                    ]
                    if len(high_severity_matches) > 0:
                        safety_net_triggered = True
                        is_bullying = True
                        pred_class = "other_cyberbullying"
                        display_category = "Other"
                        confidence = 90.0  # High confidence from slur override
                        verdict = "Cyberbullying Detected"
                        safety_net_note = "flagged via keyword safety-net, not primary model"
                
                if not safety_net_triggered:
                    pred_class = "not_cyberbullying"
                    confidence = prob_safe
                    display_category = "N/A"
                    verdict = "Not Cyberbullying"

            all_probs = {
                str(self.muril_id2label.get(i, f"Class {i}")): round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            }
            if safety_net_triggered:
                model_name = "Google MuRIL v2 (with Keyword Safety-Net Override)"
            else:
                model_name = "Google MuRIL v2 (6-Class Multilingual Transformer)"
        else:
            # Fallback for binary model v1
            prob_safe = float(probs[0]) * 100
            prob_bully = float(probs[1]) * 100
            is_bullying = (prob_bully >= 50.0)
            confidence = prob_bully if is_bullying else prob_safe
            pred_class = "other_cyberbullying" if is_bullying else "not_cyberbullying"
            display_category = "Other" if is_bullying else "N/A"
            verdict = "Cyberbullying Detected" if is_bullying else "Not Cyberbullying"
            all_probs = {
                "Safe / Non-Bully": round(prob_safe, 1),
                "Cyberbullying / Harassment": round(prob_bully, 1)
            }
            model_name = "Google MuRIL v1 (Binary Classifier)"

        # Token Attribution / Trigger Words Explainability
        if safety_net_triggered:
            from src.explainability import extract_trigger_words
            explainability = extract_trigger_words(raw_text, "Other", confidence)
            explainability["method"] = "Keyword Safety-Net Fallback"
        elif self.muril_explainer and getattr(self.muril_explainer, "explainer", None) is not None:
            explainability = self.muril_explainer.explain(raw_text, target_class=pred_class if is_bullying else None)
        else:
            from src.explainability import extract_trigger_words
            explainability = extract_trigger_words(raw_text, display_category if is_bullying else "Other", confidence)
            explainability["method"] = "Keyword-Based Trigger Detection"

        if not is_bullying and not explainability.get("highlighted_text"):
            explainability["highlighted_text"] = escape_html(raw_text)

        payload = {
            "verdict": verdict,
            "is_cyberbullying": is_bullying,
            "category": display_category,
            "confidence": round(confidence, 1),
            "model_used": model_name,
            "explainability_method": explainability.get("method", "Model-Based Token Attribution"),
            "all_probabilities": all_probs,
            "explainability": explainability
        }
        if safety_net_note:
            payload["safety_net_note"] = safety_net_note
            
        return payload

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
                "explainability_method": "None",
                "all_probabilities": {},
                "explainability": {"spans": [], "highlighted_text": "", "trigger_words": [], "top_tokens": []}
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

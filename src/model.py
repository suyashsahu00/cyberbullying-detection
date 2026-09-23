import os
import time
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional

from src.preprocessing import clean_text, detect_language, get_text_stats
from src.explainability import extract_trigger_words, escape_html

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


def classify_probabilities(
    probs: np.ndarray,
    id2label: Dict[int, str],
    method: str = "two_stage",
    safe_threshold: float = 0.50,
    raw_text: Optional[str] = None,
    use_safety_net: bool = True
) -> Dict[str, Any]:
    """
    Unified decision function used across Production, Validation, and Evaluation.
    
    Args:
        probs: 1D probability array across all classes (values sum to ~1.0 or 100.0)
        id2label: Dictionary mapping class index to label string
        method: 'two_stage' (production safety-boundary) or 'argmax' (standard baseline)
        safe_threshold: Safe threshold (0.50 means >=50% required to be considered safe)
        raw_text: Original input text (optional, for safety-net slur override)
        use_safety_net: Whether to run the Hinglish slur safety-net override
        
    Returns:
        dict with:
            pred_idx (int): Predicted class index
            pred_class (str): Predicted class label name
            confidence (float): Percentage confidence (0-100)
            is_cyberbullying (bool): Whether classified as cyberbullying
            safety_net_triggered (bool): Whether keyword safety-net overrode model
    """
    probs = np.array(probs, dtype=float)
    if np.max(probs) <= 1.0:
        probs_pct = probs * 100.0
    else:
        probs_pct = probs

    num_classes = len(probs)
    
    if method == "argmax" or num_classes != 6:
        top_idx = int(np.argmax(probs))
        pred_label = str(id2label.get(top_idx, f"Class {top_idx}"))
        is_bullying = (pred_label.lower() != "not_cyberbullying" and pred_label.lower() != "non_bully")
        return {
            "pred_idx": top_idx,
            "pred_class": pred_label,
            "confidence": float(probs_pct[top_idx]),
            "is_cyberbullying": is_bullying,
            "safety_net_triggered": False
        }

    # Two-stage production decision boundary
    not_bully_indices = [i for i, name in id2label.items() if name.lower() == "not_cyberbullying"]
    not_bully_idx = not_bully_indices[0] if not_bully_indices else 3
    
    prob_safe = float(probs_pct[not_bully_idx])
    prob_bully = 100.0 - prob_safe
    
    safe_thresh_pct = safe_threshold * 100.0 if safe_threshold <= 1.0 else safe_threshold
    is_bullying = (prob_safe < safe_thresh_pct)
    safety_net_triggered = False
    
    if is_bullying:
        bully_indices = [i for i in range(num_classes) if i != not_bully_idx]
        top_bully_idx = bully_indices[int(np.argmax([probs[i] for i in bully_indices]))]
        pred_idx = top_bully_idx
        pred_class = str(id2label[top_bully_idx])
        confidence = prob_bully

        # Option A Safety-Net: If classified as generic 'other_cyberbullying' on weak evidence,
        # ensure benign/harmless text without any trigger words isn't falsely flagged as bullying.
        if pred_class.lower() == "other_cyberbullying" and raw_text:
            from src.explainability import extract_trigger_words
            trigger_res = extract_trigger_words(raw_text, "Other", prob_safe)
            has_trigger = len(trigger_res.get("spans", [])) > 0 or len(trigger_res.get("trigger_words", [])) > 0
            
            # If no triggers exist AND the model is not overwhelmingly confident (>70% on 'other'), treat as safe
            if not has_trigger and probs_pct[top_bully_idx] < 70.0 and prob_safe >= 25.0:
                is_bullying = False
                pred_idx = not_bully_idx
                pred_class = "not_cyberbullying"
                confidence = prob_safe
    else:
        # Check hybrid safety net if safe confidence is not overwhelming (< 75%) and severe slur is present
        if use_safety_net and raw_text and prob_safe < 75.0:
            from src.explainability import extract_trigger_words, ALL_HIGH_SEVERITY_SLURS
            trigger_res = extract_trigger_words(raw_text, "Other", prob_safe)
            high_severity_matches = [
                span for span in trigger_res.get("spans", [])
                if span.get("weight", 0.0) > 0.8 and 
                (span.get("word", "").lower() in ALL_HIGH_SEVERITY_SLURS or 
                 any(w in span.get("word", "").lower() for w in ALL_HIGH_SEVERITY_SLURS) or
                 any(span.get("word", "").lower() in w for w in ALL_HIGH_SEVERITY_SLURS))
            ]
            if len(high_severity_matches) > 0:
                safety_net_triggered = True
                is_bullying = True
                
                # Determine demographic category from matched slur
                matched_span = high_severity_matches[0]
                matched_cat = str(matched_span.get("category", "Other")).lower()
                category_key_map = {
                    "age": "age",
                    "gender": "gender",
                    "ethnicity": "ethnicity",
                    "religion": "religion"
                }
                target_key = category_key_map.get(matched_cat, "other_cyberbullying")
                matched_indices = [i for i, name in id2label.items() if name.lower() == target_key]
                pred_idx = matched_indices[0] if matched_indices else 4
                pred_class = str(id2label[pred_idx])
                confidence = 90.0
        
        if not safety_net_triggered:
            pred_idx = not_bully_idx
            pred_class = "not_cyberbullying"
            confidence = prob_safe
            
    return {
        "pred_idx": pred_idx,
        "pred_class": pred_class,
        "confidence": confidence,
        "is_cyberbullying": is_bullying,
        "safety_net_triggered": safety_net_triggered
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
        """Load trained baseline pipeline with resilient fallback."""
        if os.path.exists(BASELINE_MODEL_PATH):
            try:
                with open(BASELINE_MODEL_PATH, "rb") as f:
                    self.baseline_pipeline = pickle.load(f)
                print(f" Loaded Baseline Model ({getattr(self.baseline_pipeline, 'classes_', [])})")
                return
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}")

        # Resilient fallback: build 6-class baseline in-memory
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            synthetic_samples = [
                ("Hope you have a wonderful and peaceful day ahead!", "not_cyberbullying"),
                ("Thank you so much for the helpful answer, appreciate it!", "not_cyberbullying"),
                ("Bhai call me when you reach home, safe travels.", "not_cyberbullying"),
                ("Great presentation today! Very informative slides.", "not_cyberbullying"),
                ("Shut up you senile old hag boomer grandpa!", "age"),
                ("You are too old to work here, retire already.", "age"),
                ("Go back to the kitchen and make me a sandwich bitch.", "gender"),
                ("She only got promoted because she is a golddigger thot.", "gender"),
                ("Go back to your country you illegal immigrant curryboy.", "ethnicity"),
                ("Dirty foreigner get out of here right now.", "ethnicity"),
                ("All ricebag infidels and terrorists should be banned.", "religion"),
                ("You religious extremist cultist go back to your cave.", "religion"),
                ("You are an utter idiot, kill yourself kys loser.", "other_cyberbullying"),
                ("Tu bilkul pagal chutiya aur kuttiya hai, kamina saale.", "other_cyberbullying")
            ]
            self.baseline_pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
                ("clf", LogisticRegression(C=2.0, max_iter=200))
            ])
            self.baseline_pipeline.fit([s[0] for s in synthetic_samples], [s[1] for s in synthetic_samples])
            print(" Fallback 6-class pipeline created.")
        except Exception as err:
            print(f"Error creating fallback pipeline: {err}")

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
            decision = classify_probabilities(
                probs=probs,
                id2label=self.muril_id2label,
                method="two_stage",
                safe_threshold=0.50,
                raw_text=raw_text,
                use_safety_net=True
            )
            is_bullying = decision["is_cyberbullying"]
            pred_class = decision["pred_class"]
            confidence = decision["confidence"]
            safety_net_triggered = decision["safety_net_triggered"]
            
            if is_bullying:
                display_category = CATEGORY_DISPLAY_MAP.get(pred_class.lower(), pred_class.capitalize())
                verdict = "Cyberbullying Detected"
                if safety_net_triggered:
                    safety_net_note = "flagged via keyword safety-net, not primary model"
            else:
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
            explainability = extract_trigger_words(raw_text, display_category if is_bullying else "Other", confidence)
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

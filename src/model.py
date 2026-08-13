import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text, detect_language
from src.explainability import extract_trigger_words, escape_html, TRIGGER_LEXICON

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.pkl")

# Expanded dataset of sample phrases for robust baseline training if no saved pickle exists
SYNTHETIC_DATA = [
    # Non-cyberbullying (Neutral / Friendly)
    ("Hope you have a wonderful day ahead!", "Not Cyberbullying", 0),
    ("Great presentation today! Very informative slides.", "Not Cyberbullying", 0),
    ("Bhai call me when you reach home.", "Not Cyberbullying", 0),
    ("Thanks for sharing this tutorial, super helpful.", "Not Cyberbullying", 0),
    ("Can someone recommend a good laptop for coding?", "Not Cyberbullying", 0),
    ("Awesome movie! Loved the cinematography.", "Not Cyberbullying", 0),
    ("Mera code finally compile ho gaya, yay!", "Not Cyberbullying", 0),
    ("Happy birthday! Have a great year ahead.", "Not Cyberbullying", 0),
    ("What time is the cricket match tonight?", "Not Cyberbullying", 0),
    ("Subah se baarish ho rahi hai yahan.", "Not Cyberbullying", 0),
    
    # Cyberbullying - Age
    ("Shut up you senile old hag boomer!", "Age", 1),
    ("You are too old to understand technology grandpa.", "Age", 1),
    ("Get off the internet grandma you wrinkly old fossil.", "Age", 1),
    ("Kiddo go play with your toys, toddlers can't code.", "Age", 1),
    ("Ok boomer your time is up, useless old man.", "Age", 1),
    
    # Cyberbullying - Gender
    ("Go back to the kitchen and make me a sandwich bitch.", "Gender", 1),
    ("She only got promoted because she is a golddigger thot.", "Gender", 1),
    ("Shut up you dumb hoe nobody asked your opinion.", "Gender", 1),
    ("Typical female drama, useless dish washer.", "Gender", 1),
    ("Incel loser go cry in the corner.", "Gender", 1),
    
    # Cyberbullying - Ethnicity
    ("Go back to your country you illegal immigrant scum.", "Ethnicity", 1),
    ("Curryboy stop posting online nobody likes your race.", "Ethnicity", 1),
    ("Dirty foreigner get out of here right now.", "Ethnicity", 1),
    ("Saale nigger chink go away racist idiot.", "Ethnicity", 1),
    
    # Cyberbullying - Religion
    ("You religious fanatic terrorist go back to your cult.", "Religion", 1),
    ("All ricebag infidels should be removed.", "Religion", 1),
    ("Radical extremist preaching hate as usual.", "Religion", 1),
    ("Kafir cultist stop spreading your propaganda.", "Religion", 1),
    
    # Cyberbullying - Other / General
    ("You are an utter idiot, kill yourself kys.", "Other", 1),
    ("Tu bilkul chutiya hai, kamina saala pagal.", "Other", 1),
    ("You are ugly, fat, useless garbage trash.", "Other", 1),
    ("Die already you worthless piece of shit.", "Other", 1),
    ("Bakwas band kar harami loser.", "Other", 1)
]

class CyberbullyingClassifier:
    """
    NLP Classifier wrapper with fallback rule integration and explainability support.
    """
    def __init__(self):
        self.pipeline = None
        self.categories = ["Age", "Gender", "Ethnicity", "Religion", "Other"]
        self._initialize_model()

    def _initialize_model(self):
        """Load pretrained model or train baseline synthetic model."""
        model_filepath = os.path.abspath(MODEL_PATH)
        if os.path.exists(model_filepath):
            try:
                with open(model_filepath, "rb") as f:
                    self.pipeline = pickle.load(f)
                return
            except Exception as e:
                print(f"Warning: Could not load saved model pickle ({e}). Retraining synthetic baseline...")

        # Train baseline TF-IDF + Logistic Regression
        df = pd.DataFrame(SYNTHETIC_DATA, columns=["text", "category", "label"])
        df["cleaned"] = df["text"].apply(clean_text)

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf', LogisticRegression(C=2.0, max_iter=500))
        ])
        
        self.pipeline.fit(df["cleaned"], df["label"])

    def _check_lexicon_rules(self, text: str) -> Tuple[bool, str, float]:
        """Rule-based check against trigger lexicon to bolster baseline accuracy."""
        text_lower = text.lower()
        for category, words in TRIGGER_LEXICON.items():
            for word in words:
                if f" {word} " in f" {text_lower} " or word in text_lower:
                    # High confidence match
                    return True, category, 0.88 + 0.10 * (len(word) / 15)
        return False, "Not Cyberbullying", 0.0

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Analyze input string and return complete verdict, category, confidence, language, and explainability annotations.
        """
        if not raw_text or not raw_text.strip():
            return {
                "verdict": "Not Cyberbullying",
                "is_cyberbullying": False,
                "category": "N/A",
                "confidence": 99.0,
                "language": "English",
                "original_text": "",
                "cleaned_text": "",
                "explainability": {"spans": [], "highlighted_text": "", "trigger_words": []}
            }

        cleaned = clean_text(raw_text)
        language = detect_language(raw_text)
        
        # Rule check against lexicon
        rule_hit, rule_cat, rule_conf = self._check_lexicon_rules(raw_text)
        
        # Model ML inference
        probs = self.pipeline.predict_proba([cleaned])[0]
        # Class 1 is cyberbullying, Class 0 is safe
        prob_bullying = probs[1] if len(probs) > 1 else 0.0
        prob_safe = probs[0] if len(probs) > 0 else 1.0

        if rule_hit:
            is_bullying = True
            category = rule_cat
            confidence = round(max(rule_conf, float(prob_bullying)) * 100, 1)
            confidence = min(98.5, confidence)
        elif prob_bullying >= 0.65:
            is_bullying = True
            category = "Other"
            for cat, words in TRIGGER_LEXICON.items():
                if any(w in cleaned.lower() for w in words):
                    category = cat
                    break
            confidence = round(float(prob_bullying) * 100, 1)
            confidence = min(98.5, confidence)
        else:
            is_bullying = False
            category = "N/A"
            confidence = round(float(prob_safe) * 100, 1)
            confidence = max(65.0, min(99.0, confidence))

        verdict = "Cyberbullying Detected" if is_bullying else "Not Cyberbullying"

        # Generate explainability details
        explainability = extract_trigger_words(
            raw_text, 
            category if is_bullying else "Other", 
            confidence
        )

        # If not cyberbullying, clear trigger words unless explicitly matching
        if not is_bullying:
            explainability["highlighted_text"] = escape_html(raw_text)
            explainability["trigger_words"] = []
            explainability["spans"] = []

        return {
            "verdict": verdict,
            "is_cyberbullying": is_bullying,
            "category": category,
            "confidence": confidence,
            "language": language,
            "original_text": raw_text,
            "cleaned_text": cleaned,
            "explainability": explainability
        }

# Global singleton classifier instance
_classifier_instance = None

def get_classifier() -> CyberbullyingClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = CyberbullyingClassifier()
    return _classifier_instance

if __name__ == "__main__":
    clf = get_classifier()
    print("Test 1:", clf.predict("Bhai tu bilkul pagal aur chutiya hai."))
    print("Test 2:", clf.predict("Have a fantastic day everyone!"))

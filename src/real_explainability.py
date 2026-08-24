import os
import re
import html
from typing import List, Dict, Any, Tuple, Optional
import torch

try:
    from transformers_interpret import SequenceClassificationExplainer
    HAS_TRANSFORMERS_INTERPRET = True
except ImportError:
    HAS_TRANSFORMERS_INTERPRET = False

def escape_html(text: str) -> str:
    """Escape special HTML characters to prevent XSS."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))

class MuRILExplainer:
    """
    Real Gradient-Based Token Attribution Explainer for Google MuRIL Sequence Classification.
    Uses SequenceClassificationExplainer from transformers-interpret (Captum under the hood).
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = None
        
        if HAS_TRANSFORMERS_INTERPRET and model is not None and tokenizer is not None:
            try:
                self.explainer = SequenceClassificationExplainer(model, tokenizer)
            except Exception as e:
                print(f"Warning: Could not initialize SequenceClassificationExplainer: {e}")

    def explain(self, text: str, target_class: Optional[str] = None, top_k_threshold: float = 0.15) -> Dict[str, Any]:
        """
        Computes token-level gradient attribution for the input text.
        Returns:
            - word_attributions: raw [(token, score), ...] list
            - top_tokens: list of (word, score) with highest positive attributions
            - highlighted_text: HTML string with tokens colored by gradient attribution
            - spans: list of token span dictionaries
        """
        if not text or not text.strip():
            return {
                "method": "Model-Based Token Attribution (Gradient)",
                "raw_attributions": [],
                "top_tokens": [],
                "highlighted_text": "",
                "spans": []
            }

        if self.explainer is None:
            # Fallback if explainer failed to initialize
            return {
                "method": "Model-Based Token Attribution (Gradient - Unavailable)",
                "raw_attributions": [],
                "top_tokens": [],
                "highlighted_text": escape_html(text),
                "spans": []
            }

        try:
            # Run gradient attribution pass
            attributions = self.explainer(text, class_name=target_class)
            # Filter special tokens
            filtered_attrs = [
                (tok, float(score)) for tok, score in attributions 
                if tok not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]
            ]

            # Reconstruct whole words from WordPiece subwords (e.g. 'fucker' from 'fuck' + '##er')
            merged_words = []
            curr_word = ""
            curr_scores = []
            
            for tok, score in filtered_attrs:
                if tok.startswith("##"):
                    curr_word += tok[2:]
                    curr_scores.append(score)
                else:
                    if curr_word:
                        merged_words.append((curr_word, max(curr_scores)))
                    curr_word = tok
                    curr_scores = [score]
            if curr_word:
                merged_words.append((curr_word, max(curr_scores)))

            # Sort positive contributing tokens
            positive_tokens = [(w, s) for w, s in merged_words if s > 0]
            positive_tokens.sort(key=lambda x: x[1], reverse=True)

            # Build HTML highlighting using gradient intensities
            # Find max score for relative opacity scaling
            max_score = max([s for _, s in merged_words if s > 0], default=1.0)
            if max_score <= 0:
                max_score = 1.0

            # Reconstruct spans in text
            spans = []
            highlighted_tokens = []
            
            for word, score in merged_words:
                if score > top_k_threshold:
                    # Intensity between 0.25 and 1.0
                    intensity = min(1.0, max(0.25, score / max_score))
                    rgba = f"rgba(230, 57, 70, {intensity:.2f})"
                    highlighted_tokens.append(
                        f'<mark class="token-attribution" style="background-color: {rgba}; padding: 2px 4px; border-radius: 4px; font-weight: 600;" title="Attribution: +{score:.3f}">{escape_html(word)}</mark>'
                    )
                    spans.append({
                        "token": word,
                        "attribution_score": round(score, 4),
                        "importance_weight": round(intensity, 2)
                    })
                else:
                    highlighted_tokens.append(escape_html(word))

            # Assemble highlighted HTML text
            highlighted_html = " ".join(highlighted_tokens)

            return {
                "method": "Model-Based Token Attribution (Gradient)",
                "raw_attributions": [(tok, round(score, 4)) for tok, score in filtered_attrs],
                "top_tokens": [(w, round(s, 4)) for w, s in positive_tokens[:5]],
                "highlighted_text": highlighted_html,
                "spans": spans
            }

        except Exception as e:
            print(f"Error computing gradient attributions: {e}")
            return {
                "method": "Model-Based Token Attribution (Error)",
                "raw_attributions": [],
                "top_tokens": [],
                "highlighted_text": escape_html(text),
                "spans": [],
                "error": str(e)
            }

# Module level helper
def get_explainer(model, tokenizer) -> MuRILExplainer:
    return MuRILExplainer(model, tokenizer)

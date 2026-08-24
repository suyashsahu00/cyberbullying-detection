import re
from typing import List, Dict, Any, Tuple

# Categories of trigger words for fallback/explainability highlighting across Age, Gender, Ethnicity, Religion, and General cyberbullying.
TRIGGER_LEXICON = {
    "Age": [
        "boomer", "oldhag", "old hag", "senile", "wrinkly", "toddler", "kiddo", "grandma", 
        "grandpa", "boomers", "too old", "expiring", "grandma", "dinosaur", "fossil"
    ],
    "Gender": [
        "bitch", "whore", "slut", "cunt", "kitchen", "make me a sandwich", "femcel",
        "incel", "golddigger", "hoe", "thot", "dish washer", "pussy", "simp", "misogynist"
    ],
    "Ethnicity": [
        "nigger", "nigga", "chink", "spic", "curryboy", "curry", "go back to your country",
        "foreigner", "immigrant", "illegal", "racist", "goy", "beaner", "wetback", "gypsy"
    ],
    "Religion": [
        "terrorist", "kafir", "infidel", "jihadist", "extremist", "ricebag", "heathen",
        "cultist", "paki", "islamophobe", "anti-semite", "zealot", "radical"
    ],
    "Other": [
        "ugly", "fat", "loser", "kill yourself", "kys", "die", "stupid", "idiot",
        "dumb", "useless", "trash", "garbage", "chutiya", "saale", "kamina", "harami",
        "bakwas", "fuck", "shit", "retard", "scum", "pig", "freak", "disgusting",
        "pagal", "kuttiya", "gandu", "kamine", "madarchod", "bhosdike", "randi", "kutte",
        "bhosdiwala", "gaand", "phaad", "behenchod"
    ]
}

def extract_trigger_words(text: str, category: str, confidence: float) -> Dict[str, Any]:
    """
    Extract trigger words and return character spans and highlighted HTML/annotated tokens.
    """
    if not text:
        return {"spans": [], "highlighted_text": "", "trigger_words": []}

    text_lower = text.lower()
    matches = []
    
    # 1. Search category specific triggers first, then general triggers
    search_categories = [category] if category in TRIGGER_LEXICON else []
    for cat in TRIGGER_LEXICON:
        if cat not in search_categories:
            search_categories.append(cat)
            
    found_words = set()
    
    for cat in search_categories:
        keywords = TRIGGER_LEXICON.get(cat, [])
        for word in keywords:
            # Build flexible regex allowing character elongation (e.g., 'biiiitch', 'stuuupid', 'chuuutiya')
            escaped_chars = [r'\s+' if c.isspace() else (re.escape(c) + '+' if c.isalpha() else re.escape(c)) for c in word]
            flexible_pattern = r'\b' + ''.join(escaped_chars) + r'\b'
            
            pattern = re.compile(flexible_pattern, re.IGNORECASE)
            for m in pattern.finditer(text):
                start, end = m.span()
                matched_str = text[start:end]
                found_words.add(matched_str)
                matches.append({
                    "start": start,
                    "end": end,
                    "word": matched_str,
                    "category": cat,
                    "weight": round(min(0.95, 0.6 + 0.35 * (len(matched_str) / 10)), 2)
                })

    # Deduplicate overlapping spans (keep longest / first)
    matches = sorted(matches, key=lambda x: (x["start"], -(x["end"] - x["start"])))
    filtered_spans = []
    last_end = -1
    for m in matches:
        if m["start"] >= last_end:
            filtered_spans.append(m)
            last_end = m["end"]

    # Reconstruct text with html highlighting for frontend rendering
    highlighted_parts = []
    idx = 0
    for span in filtered_spans:
        if span["start"] > idx:
            highlighted_parts.append(escape_html(text[idx:span["start"]]))
        
        trigger_token = text[span["start"]:span["end"]]
        highlighted_parts.append(
            f'<mark class="trigger-highlight" data-category="{span["category"]}" title="Trigger: {span["category"]}">{escape_html(trigger_token)}</mark>'
        )
        idx = span["end"]

    if idx < len(text):
        highlighted_parts.append(escape_html(text[idx:]))

    highlighted_text = "".join(highlighted_parts)

    return {
        "spans": filtered_spans,
        "highlighted_text": highlighted_text,
        "trigger_words": list(found_words)
    }

def escape_html(text: str) -> str:
    """Escape special HTML characters to prevent XSS."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))

if __name__ == "__main__":
    sample = "You are a stupid loser boomer go back to your country!"
    res = extract_trigger_words(sample, "Ethnicity", 0.92)
    print("Highlighted:", res["highlighted_text"])
    print("Triggers:", res["trigger_words"])

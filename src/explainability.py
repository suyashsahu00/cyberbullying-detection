import re
from typing import Dict, Any

# Categories of trigger words for fallback/explainability highlighting across Age, Gender, Ethnicity, Religion, and General cyberbullying.
TRIGGER_LEXICON = {
    "Age": [
        "boomer", "oldhag", "old hag", "senile", "wrinkly", "toddler", "kiddo", "grandma", 
        "grandpa", "boomers", "too old", "expiring", "grandma", "dinosaur", "fossil",
        "die already", "buddhe log ghar pe", "buddhe logon ko ghar pe", "बूढ़े लोग घर पे", "बूढ़े लोगों को घर पे"
    ],
    "Gender": [
        "bitch", "whore", "slut", "cunt", "kitchen", "make me a sandwich", "femcel",
        "incel", "golddigger", "hoe", "thot", "dish washer", "pussy", "simp", "misogynist",
        "belong in the kitchen", "belong only in the kitchen", "kitchen ke liye", "किचन", "किचन के लिए", "raped", "rape"
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
        "bhosdiwala", "gaand", "phaad", "behenchod", "chudail", "rand",
        "chappal khol", "chappal nikaal", "chappal nikal", "jaa mar", "ja mar", "mar jaa", "jaake mar",
        "kalank", "lode", "laude", "worthless animals", "bloodline", "dimaag mat use karna", "dimaag use mat",
        "चूतिया", "साले", "कमीना", "हरामी", "कुतिया", "गांडू", "कमीने", "मादरचोद",
        "भोसड़ीके", "रंडी", "भोसड़ीवाला", "गांड", "बहनचोद", "चुड़ैल", "कुत्ते", "नीच",
        "जा मर जा", "मर जा", "जाके मर", "कलंक", "चप्पल खोल", "चप्पल निकाल", "लोड़े", "लौड़े", "दिमाग मत यूज़ करना", "दिमाग यूज़ मत"
    ]
}

# List of highly toxic Hindi/Hinglish slurs for safety-net override
HIGH_SEVERITY_HINGLISH = {
    "chutiya", "saale", "kamina", "harami", "kuttiya", "gandu", "kamine", "madarchod", 
    "bhosdike", "randi", "rand", "bhosdiwala", "gaand", "behenchod", "chudail",
    "chappal khol", "chappal nikaal", "chappal nikal", "jaa mar", "ja mar", "mar jaa", "jaake mar",
    "kalank", "lode", "laude", "buddhe log ghar pe", "buddhe logon ko ghar pe", "kitchen ke liye"
}

# List of severe Devanagari Hindi slurs for safety-net override
HIGH_SEVERITY_DEVANAGARI = {
    "चूतिया", "साले", "कमीना", "हरामी", "कुतिया", "गांडू", "कमीने", "मादरचोद",
    "भोसड़ीके", "रंडी", "भोसड़ीवाला", "गांड", "बहनचोद", "चुड़ैल", "कुत्ते", "नीच",
    "जा मर जा", "मर जा", "जाके मर", "कलंक", "चप्पल खोल", "चप्पल निकाल", "लोड़े", "लौड़े", "बूढ़े लोग घर पे", "बूढ़े लोगों को घर पे", "किचन के लिए"
}

# List of severe English slurs, targeted profanity, and harassment triggers for safety-net override
HIGH_SEVERITY_ENGLISH = {
    "whore", "slut", "cunt", "bitch", "hoe", "thot",
    "nigger", "nigga", "chink", "spic", "faggot", "beaner", "wetback", "curryboy",
    "ricebag", "infidel", "kafir",
    "kill yourself", "kys", "motherfucker", "suck my dick", "go fuck yourself", "suck my cock", "fuck off",
    "raped", "die alone", "worthless animals", "bloodline", "die already", "belong in the kitchen", "belong only in the kitchen"
}

ALL_HIGH_SEVERITY_SLURS = HIGH_SEVERITY_HINGLISH | HIGH_SEVERITY_ENGLISH | HIGH_SEVERITY_DEVANAGARI

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
            has_non_ascii = any(ord(c) > 127 for c in word)
            if has_non_ascii:
                pattern = re.compile(r'(?<!\w)' + re.escape(word) + r'(?!\w)', re.IGNORECASE)
            else:
                escaped_chars = [r'\s+' if c.isspace() else (re.escape(c) + '+' if c.isalpha() else re.escape(c)) for c in word]
                flexible_pattern = r'\b' + ''.join(escaped_chars) + r'\b'
                pattern = re.compile(flexible_pattern, re.IGNORECASE)

            for m in pattern.finditer(text):
                start, end = m.span()
                matched_str = text[start:end]
                found_words.add(matched_str)
                
                # Check if it is a high-severity slur (Hinglish or English) to boost weight for the safety-net
                word_lower = matched_str.lower()
                is_high_severity = (word_lower in ALL_HIGH_SEVERITY_SLURS or 
                                    any(w in word_lower for w in ALL_HIGH_SEVERITY_SLURS) or
                                    any(word_lower in w for w in ALL_HIGH_SEVERITY_SLURS))
                weight = 0.90 if is_high_severity else round(min(0.95, 0.6 + 0.35 * (len(matched_str) / 10)), 2)
                
                matches.append({
                    "start": start,
                    "end": end,
                    "word": matched_str,
                    "category": cat,
                    "weight": weight
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

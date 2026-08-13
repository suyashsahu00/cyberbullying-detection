import re
import string
from typing import Tuple, Dict, Any, List

# Common Hinglish (Hindi in Latin script) marker words for heuristic language detection
HINGLISH_KEYWORDS = {
    'bhai', 'yaar', 'tera', 'teri', 'tere', 'mera', 'meri', 'mere', 'kya', 'hai', 'hain',
    'karo', 'karna', 'raha', 'rahi', 'rahe', 'nahi', 'nhi', 'mat', 'hoye', 'bol', 'rha',
    'rhi', 'ab', 'tak', 'bhi', 'kuch', 'kaise', 'kab', 'aur', 'wale', 'wali', 'wala',
    'sab', 'saale', 'kamine', 'kamina', 'pagal', 'chutiya', 'bc', 'mc', 'gaali', 'harami',
    'bakwas', 'band', 'thik', 'sahi', 'par', 'pe', 'se', 'ko', 'ne', 'ka', 'ke', 'ki'
}

def clean_text(text: str) -> str:
    """
    Clean and normalize input text for cyberbullying analysis.
    - Preserves case sensitivity where relevant for trigger matching, but normalizes whitespace and URLs.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Normalize user mentions (@username -> @user)
    text = re.sub(r'@\w+', '@user', text)
    
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_language(text: str) -> str:
    """
    Detect whether text is predominantly English or Hinglish.
    Returns: 'English' or 'Hinglish'
    """
    cleaned = clean_text(text).lower()
    words = re.findall(r'\b[a-z]+\b', cleaned)
    
    if not words:
        return 'English'

    hinglish_count = sum(1 for word in words if word in HINGLISH_KEYWORDS)
    ratio = hinglish_count / len(words)
    
    # If >= 15% of words or 2+ distinct markers are Hinglish keywords, tag as Hinglish
    if ratio >= 0.12 or hinglish_count >= 2:
        return 'Hinglish'
    
    return 'English'

def get_text_stats(text: str) -> Dict[str, Any]:
    """
    Extract token length, word count, character count, and language badge.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    return {
        "cleaned_text": cleaned,
        "char_count": len(cleaned),
        "word_count": len(words),
        "language": detect_language(cleaned)
    }

if __name__ == "__main__":
    sample_en = "You are so stupid and nobody likes your tweets."
    sample_hi = "Bhai tu bilkul pagal hai mat bol yahan."
    
    print("EN Test:", get_text_stats(sample_en))
    print("HI Test:", get_text_stats(sample_hi))

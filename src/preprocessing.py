import html
import os
import re
import string
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Common Hinglish (Hindi in Latin script) marker words for heuristic language detection
HINGLISH_KEYWORDS = {
    'bhai', 'yaar', 'tera', 'teri', 'tere', 'mera', 'meri', 'mere', 'kya', 'hai', 'hain',
    'karo', 'karna', 'raha', 'rahi', 'rahe', 'nahi', 'nhi', 'mat', 'hoye', 'bol', 'rha',
    'rhi', 'ab', 'tak', 'bhi', 'kuch', 'kaise', 'kab', 'aur', 'wale', 'wali', 'wala',
    'sab', 'saale', 'kamine', 'kamina', 'pagal', 'chutiya', 'bc', 'mc', 'gaali', 'harami',
    'bakwas', 'band', 'thik', 'sahi', 'par', 'pe', 'se', 'ko', 'ne', 'ka', 'ke', 'ki',
    'kuttiya', 'randi', 'gandu', 'chut', 'moti', 'gadhi', 'lund', 'madarchod', 'bhosdike',
    'bhosdiwala', 'gaand', 'phaad', 'behenchod'
}

def clean_text(
    text: str,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    decode_html: bool = True,
    normalize_whitespace: bool = True,
    strip_brackets: bool = False
) -> str:
    """
    Standardizes and normalizes social media text for Transformer & Classical NLP models.
    
    Parameters:
    -----------
    text: Raw string
    replace_urls: If True, replaces http/https/www URLs with '<URL>'
    replace_mentions: If True, replaces @username with '<USER>'
    decode_html: If True, converts HTML entities (&amp; -> &, &lt; -> <)
    normalize_whitespace: If True, collapses tabs/newlines/multiple spaces into single space
    strip_brackets: If True, removes curly braces used in rationale annotations
    """
    if not text or not isinstance(text, str):
        return ""

    # Decode HTML entities
    if decode_html:
        text = html.unescape(text)

    # Optional: strip curly braces if cleaning rationale-annotated text
    if strip_brackets:
        text = text.replace('{', '').replace('}', '')

    # Normalize URLs
    if replace_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)

    # Normalize user mentions
    if replace_mentions:
        text = re.sub(r'@\w+', '<USER>', text)

    # Remove non-printable / control characters (preserve emojis and standard unicode)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize multiple whitespace characters
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_rationales(text: str) -> List[str]:
    """
    Extracts word-level rationale tokens enclosed in or prefixed by curly braces '{word'
    from BullyExplain dataset annotations.
    """
    if not text or not isinstance(text, str):
        return []
    matches = re.findall(r'\{([a-zA-Z0-9_\u0900-\u097F]+)', text)
    return [m.lower().strip() for m in matches if len(m.strip()) > 1]

def detect_language(text: str) -> str:
    """
    Detect whether text is predominantly English or Hinglish (Hindi in Latin/Devanagari script).
    Returns: 'English', 'Hinglish', or 'Hindi'
    """
    if not text or not isinstance(text, str):
        return 'English'
        
    # Check for Devanagari script presence
    if re.search(r'[\u0900-\u097F]', text):
        return 'Hindi'

    cleaned = text.lower()
    words = re.findall(r'\b[a-z]+\b', cleaned)
    
    if not words:
        return 'English'

    hinglish_count = sum(1 for word in words if word in HINGLISH_KEYWORDS)
    ratio = hinglish_count / len(words)
    
    # If >= 10% of words or 2+ distinct markers are Hinglish keywords, tag as Hinglish
    if ratio >= 0.10 or hinglish_count >= 2:
        return 'Hinglish'
    
    return 'English'

def get_text_stats(text: str) -> Dict[str, Any]:
    """
    Extract token length, word count, character count, uppercase ratio, and language tag.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / (len(text) + 1e-5)
    
    return {
        "cleaned_text": cleaned,
        "char_count": len(cleaned),
        "word_count": len(words),
        "caps_ratio": round(caps_ratio, 4),
        "language": detect_language(text)
    }

def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into Train, Validation, and Test sets with stratification on target_col.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    # First split: train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df[target_col]
    )
    
    # Second split: val vs test
    val_rel_ratio = val_ratio / temp_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_rel_ratio),
        random_state=random_state,
        stratify=df_temp[target_col]
    )
    
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def save_dataset_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    prefix: str,
    output_dir: str
) -> Dict[str, str]:
    """
    Saves Train/Val/Test splits to disk in both CSV and Parquet formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    
    splits = {
        f"{prefix}_train": df_train,
        f"{prefix}_val": df_val,
        f"{prefix}_test": df_test
    }
    
    for name, split_df in splits.items():
        csv_path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(csv_path, index=False, encoding='utf-8')
        saved_paths[f"{name}_csv"] = csv_path
        
        try:
            parquet_path = os.path.join(output_dir, f"{name}.parquet")
            split_df.to_parquet(parquet_path, index=False)
            saved_paths[f"{name}_parquet"] = parquet_path
        except Exception as e:
            # Fallback if parquet engine is not yet available
            pass

    return saved_paths

if __name__ == "__main__":
    sample_en = "You are so stupid @johndoe and nobody likes your tweets! https://toxic.com"
    sample_hi = "Bhai tu bilkul kuttiya aur pagal hai mat bol yahan."
    
    print("EN Test Stats:", get_text_stats(sample_en))
    print("HI Test Stats:", get_text_stats(sample_hi))

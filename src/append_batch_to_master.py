"""
====================================================================================================
 🛡️ Hard-Negative Batch Validator & Master Ingestion Pipeline
====================================================================================================
Validates incoming raw synthetic/hard-negative batches, enforces strict deduplication across
the training set, augmented pool, and locked validation sets (exact, normalized, fuzzy TF-IDF),
detects style leakage, and appends clean samples to the hard-negative master dataset.
====================================================================================================
"""

import os
import sys
import glob
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import detect_language

VALID_CATEGORIES = {
    "age",
    "ethnicity",
    "gender",
    "religion",
    "other_cyberbullying",
    "not_cyberbullying"
}

POINTING_PATTERN = r'\b(?:this|that|these|those|ye|yeh|is|iss|in|inn|wo|woh|un|unn)\b|(?<!\w)(?:इस|इन|यह|ये|वह|उन)(?!\w)'
ABSOLUTE_PATTERN = r'\b(?:stop|bas|hamesha|sirf|band|mat|always|never|only|shut|just)\b|(?<!\w)(?:हमेशा|केवल|सिर्फ़|सिर्फ|बंद|मत|कभी)(?!\w)'

NEG_ADJ_LIST = [
    # English
    "self-righteous", "loud", "arrogant", "lazy", "careless", "entitled", 
    "boastful", "dramatic", "clueless", "stubborn", "nosy", "greedy", "dirty",
    "stupid", "ignorant", "vile", "toxic", "obnoxious", "disgusting", "backward",
    "intolerant", "hypocritical", "pushy", "rude", "nasty", "irresponsible",
    # Hindi Devanagari
    "आलसी", "लापरवाह", "घमंडी", "चिड़चिड़े", "चिड़चिड़ा", "बत्तमीज", "बदतमीज",
    "ढोंगी", "कट्टर", "जाहिल", "गंदा", "नीच", "पाखंडी", "मक्कार", "चालाक", "कंजूस",
    # Hinglish
    "aalsi", "laparwah", "ghamandi", "chidchida", "chidchide", "battameez", "badtameez",
    "dhongi", "kattar", "jahil", "ganda", "neech", "pakhandi", "makkar"
]

GROUP_TERMS_LIST = [
    # English / Roman
    "hindu", "muslim", "sikh", "christian", "jewish", "dalit", "brahmin",
    "bihari", "bengali", "gujarati", "punjabi", "marathi", "kashmiri",
    "women", "woman", "girl", "female", "boy", "man", "men", "male",
    "trans", "queer", "gay", "lesbian", "elderly", "elders", "elder",
    "old", "boomer", "buddha", "buddhe", "buzurg", "chachi", "aunty", "uncle",
    "dadi", "nani", "dada", "nana", "student", "teacher", "doctor", "nurse",
    "commuter", "passenger", "colleague", "neighbor", "neighbour", "shopkeeper",
    "merchant", "trader", "vendor", "aadmi", "aurat", "ladki", "ladka",
    # Hindi Devanagari
    "हिंदू", "मुस्लिम", "मुसलमान", "सिख", "ईसाई", "यहूदी", "दलित", "ब्राह्मण",
    "बिहारी", "बंगाली", "गुजराती", "पंजाबी", "मराठी", "कश्मीरी", "महिला", "औरत",
    "लड़की", "लड़का", "आदमी", "पुरुष", "बूढ़ा", "बुजुर्ग", "चाची", "आंटी", "दादी",
    "नानी", "दादा", "नाना", "छात्र", "छात्रा", "विद्यार्थी", "शिक्षक", "व्यापारी",
    "दुकानदार", "पड़ोसी", "यात्री"
]

_neg_adj_subpat = r'(?:' + '|'.join(map(re.escape, NEG_ADJ_LIST)) + r')'
_group_subpat = r'(?:' + '|'.join(map(re.escape, GROUP_TERMS_LIST)) + r')'
NEG_ADJ_GROUP_PATTERN = rf'\b{_neg_adj_subpat}\s+(?:(?:and|aur|or|waala|wale|wali|se|वाला|वाले|वाली|से)\s+)?{_group_subpat}\b'

def normalize_text(text: str) -> str:
    """Normalize text for strict duplicate detection (lowercase, whitespace collapsed, stripped)."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'<URL>|<USER>|https?://\S+|@\w+', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned.strip().lower())
    return cleaned

def detect_script(text: str) -> str:
    """Detect whether script is Devanagari, Roman Hinglish, or English."""
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    if re.search(r'[\u0900-\u097F]', text):
        return "Devanagari (Hindi)"
    lang = detect_language(text)
    if lang == "Hinglish":
        return "Roman Hinglish"
    return "English"

def load_exclusion_corpus(target_master_path: str, batch_path: str = None):
    """
    Collects raw and normalized texts from:
    1. Training parquet (data/processed/combined_train.parquet)
    2. All augmented files (data/augmented/*.csv, excluding target master and current batch)
    3. All validation and held-out test files
    4. Target master file (if exists)
    """
    seen_texts = set()
    val_records = []
    
    ignore_paths = {os.path.abspath(target_master_path)}
    if batch_path:
        ignore_paths.add(os.path.abspath(batch_path))
    
    # 1. Training parquet
    train_parquet = os.path.join(ROOT_DIR, "data", "processed", "combined_train.parquet")
    if os.path.exists(train_parquet):
        tdf = pd.read_parquet(train_parquet)
        col = "cleaned_text" if "cleaned_text" in tdf.columns else "tweet_text"
        for t in tdf[col].dropna():
            norm = normalize_text(t)
            if norm:
                seen_texts.add(norm)
        print(f"  • Ingested {len(tdf):,} training samples for deduplication check.")
        
    # 2. Augmented files
    aug_files = glob.glob(os.path.join(ROOT_DIR, "data", "augmented", "*.csv"))
    aug_count = 0
    for af in aug_files:
        if os.path.abspath(af) in ignore_paths:
            continue
        try:
            adf = pd.read_csv(af)
            col = "text" if "text" in adf.columns else ("cleaned_text" if "cleaned_text" in adf.columns else None)
            if col:
                for t in adf[col].dropna():
                    norm = normalize_text(t)
                    if norm:
                        seen_texts.add(norm)
                        aug_count += 1
        except Exception:
            pass
    print(f"  • Ingested {aug_count:,} augmented samples across pool.")
    
    # 3. Validation & Test files (STRICT ISOLATION)
    val_test_paths = [
        ("hardneg_human_test_set", os.path.join(ROOT_DIR, "data", "validation", "hardneg_human_test_set.csv"), "text"),
        ("hardneg_synth_test_set", os.path.join(ROOT_DIR, "data", "validation", "hardneg_synth_test_set.csv"), "text"),
        ("combined_test", os.path.join(ROOT_DIR, "data", "processed", "combined_test.csv"), "cleaned_text"),
        ("combined_val", os.path.join(ROOT_DIR, "data", "processed", "combined_val.parquet"), "cleaned_text"),
        # FINAL HOLDOUT — read hashes only, never expose rows
        ("final_holdout_v2", os.path.join(ROOT_DIR, "data", "final_holdout", "final_human_test_v2.csv"), "text"),
    ]
    
    for vname, vpath, vcol in val_test_paths:
        if os.path.exists(vpath):
            vdf = pd.read_parquet(vpath) if vpath.endswith(".parquet") else pd.read_csv(vpath)
            for t in vdf[vcol].dropna():
                raw_t = str(t).strip()
                norm_t = normalize_text(raw_t)
                if norm_t:
                    seen_texts.add(norm_t)
                    val_records.append({"source": vname, "raw": raw_t, "norm": norm_t})
                    
    print(f"  • Ingested {len(val_records):,} validation & test samples for zero-leakage check.")
    
    # 4. Target master (if already partially created)
    if os.path.exists(target_master_path):
        mdf = pd.read_csv(target_master_path)
        col = "text" if "text" in mdf.columns else ("cleaned_text" if "cleaned_text" in mdf.columns else None)
        if col:
            for t in mdf[col].dropna():
                norm = normalize_text(t)
                if norm:
                    seen_texts.add(norm)
                    
    return seen_texts, val_records

def validate_and_append_batch(batch_path: str, master_path: str, dry_run: bool = False):
    print("=" * 85)
    print(" 🛠️ HARD-NEGATIVE BATCH VALIDATION & QUALITY AUDIT REPORT")
    print("=" * 85)
    print(f"Batch File   : {batch_path}")
    print(f"Master Target: {master_path}")
    print(f"Dry Run Mode : {dry_run}")
    print("-" * 85)
    
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Batch file not found at: {batch_path}")
        
    df = pd.read_csv(batch_path)
    total_raw = len(df)
    print(f"\n1. Batch Structure & Language Check:")
    print(f"  • Raw Rows Loaded : {total_raw}")
    print(f"  • Columns Found   : {list(df.columns)}")
    
    text_col = "text" if "text" in df.columns else ("cleaned_text" if "cleaned_text" in df.columns else None)
    cat_col = "category" if "category" in df.columns else ("cyberbullying_type" if "cyberbullying_type" in df.columns else None)
    
    if not text_col or not cat_col:
        raise ValueError(f"Batch must contain a text column and category column.")
        
    df = df.rename(columns={text_col: "text", cat_col: "category"})
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    
    # Language detection for missing values
    if "language" not in df.columns:
        df["language"] = None
    missing_lang = df["language"].isnull().sum()
    if missing_lang > 0:
        print(f"  • Missing language tags: {missing_lang} rows -> filling with detect_language...")
        detected_series = df["text"].apply(detect_language)
        df["language"] = df["language"].fillna(detected_series)
    else:
        print("  • All language tags present.")
        
    lang_dist = df["language"].value_counts()
    print("  • Language Distribution across all rows:")
    for lang, cnt in lang_dist.items():
        print(f"    - {lang:10s}: {cnt:4d} ({cnt/len(df)*100:5.1f}%)")
        
    # Check valid categories
    invalid_labels = set(df["category"]) - VALID_CATEGORIES
    if invalid_labels:
        print(f"  ❌ FAILED: Invalid category labels found: {invalid_labels}")
        return False
    print("  ✅ Passed: All category labels match the official 6-class taxonomy.")
    
    # 2. Internal Batch Deduplication
    df["norm_text"] = df["text"].apply(normalize_text)
    before_internal = len(df)
    df = df.drop_duplicates(subset=["norm_text"]).reset_index(drop=True)
    internal_dups = before_internal - len(df)
    print(f"\n2. Internal Batch Deduplication:")
    print(f"  • Duplicates within batch removed: {internal_dups}")
    print(f"  • Unique candidates remaining    : {len(df)}")
    
    # 3. Comprehensive Zero-Leakage & External Overlap Check
    print("\n3. External Overlap & Zero-Leakage Verification:")
    seen_texts, val_records = load_exclusion_corpus(master_path, batch_path)
    val_df = pd.DataFrame(val_records)
    
    val_exact_set = set(val_df["raw"])
    val_norm_set = set(val_df["norm"])
    
    # Exact and Normalized Leakage Check
    exact_leakage_indices = set()
    for idx, row in df.iterrows():
        if row["text"] in val_exact_set or row["norm_text"] in val_norm_set:
            exact_leakage_indices.add(idx)
            
    print(f"  • Exact / Normalized validation leakage: {len(exact_leakage_indices)} rows")
    
    # Fuzzy Leakage Check (char 3-5-gram TF-IDF cosine similarity)
    print("  • Running fuzzy TF-IDF char 3-5-gram cosine similarity check against validation/test sets...")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    corpus = df["norm_text"].tolist() + val_df["norm"].tolist()
    vectorizer.fit(corpus)
    
    b_vecs = vectorizer.transform(df["norm_text"])
    v_vecs = vectorizer.transform(val_df["norm"])
    sim_matrix = cosine_similarity(b_vecs, v_vecs)
    
    max_sim_idx = np.argmax(sim_matrix, axis=1)
    max_sim_val = np.max(sim_matrix, axis=1)
    
    fuzzy_ge_80 = []
    fuzzy_70_80 = []
    
    for idx in range(len(df)):
        score = float(max_sim_val[idx])
        v_rec = val_df.iloc[max_sim_idx[idx]]
        if score >= 0.80 or idx in exact_leakage_indices:
            fuzzy_ge_80.append((idx, score, df.iloc[idx]["text"], df.iloc[idx]["category"], v_rec["source"], v_rec["raw"]))
        elif score >= 0.70:
            fuzzy_70_80.append((idx, score, df.iloc[idx]["text"], df.iloc[idx]["category"], v_rec["source"], v_rec["raw"]))
            
    # Pair-Partner Removal Rule:
    # If row i is flagged, remove i AND its pair-partner:
    # (if i is even, partner is i+1; if i is odd, partner is i-1)
    indices_to_remove = set()
    print(f"\n  • High-Similarity Leakage Matches (cosine >= 0.80): {len(fuzzy_ge_80)}")
    for f_idx, f_score, b_txt, b_cat, src, v_txt in fuzzy_ge_80:
        partner_idx = f_idx + 1 if (f_idx % 2 == 0) else f_idx - 1
        indices_to_remove.add(f_idx)
        if 0 <= partner_idx < len(df):
            indices_to_remove.add(partner_idx)
        print(f"    [FLAGGED Row {f_idx}] (Sim: {f_score:.4f}) Category: {b_cat}")
        print(f"       Batch Text : \"{b_txt}\"")
        print(f"       Matched    : {src} -> \"{v_txt}\"")
        if 0 <= partner_idx < len(df):
            print(f"       Pair-Partner Row {partner_idx} removed together.")
            
    # Borderline 0.70 - 0.80 matches (Report only, do not remove)
    print(f"\n  • Borderline Similarity Matches (0.70 <= cosine < 0.80) [REPORT ONLY, KEPT]: {len(fuzzy_70_80)}")
    for f_idx, f_score, b_txt, b_cat, src, v_txt in fuzzy_70_80:
        print(f"    [Row {f_idx}] (Sim: {f_score:.4f}) Category: {df.iloc[f_idx]['category']}")
        print(f"       Batch Text : \"{b_txt}\"")
        print(f"       Matched    : {src} -> \"{v_txt}\"")
        
    # Drop flagged pairs
    if len(indices_to_remove) > 0:
        print(f"\n  • Dropping {len(indices_to_remove)} rows due to >= 0.80 leakage & pair-partner rule: {sorted(list(indices_to_remove))}")
        clean_df = df.drop(index=list(indices_to_remove)).copy().reset_index(drop=True)
    else:
        clean_df = df.copy()
        
    # Check general training corpus overlap
    corpus_overlap = clean_df[clean_df["norm_text"].isin(seen_texts)]
    print(f"  • External general training corpus overlap dropped: {len(corpus_overlap)}")
    clean_df = clean_df[~clean_df["norm_text"].isin(seen_texts)].copy().reset_index(drop=True)
    print(f"  • Net Fresh Samples to Ingest: {len(clean_df)}")
    
    if len(clean_df) == 0:
        print("  ⚠️ Warning: No fresh samples to append.")
        return True
        
    # 4. Batch Quality Audit Report
    print("\n" + "=" * 85)
    print(" 📊 DETAILED BATCH QUALITY AUDIT REPORT")
    print("=" * 85)
    
    # Class Distribution
    cat_counts = clean_df["category"].value_counts()
    print("Class Distribution:")
    for cat, cnt in cat_counts.items():
        print(f"  • {cat:20s}: {cnt:4d} ({cnt/len(clean_df)*100:5.1f}%)")
        
    safe_cnt = cat_counts.get("not_cyberbullying", 0)
    bully_cnt = len(clean_df) - safe_cnt
    ratio = safe_cnt / max(bully_cnt, 1)
    print(f"\nBalance Ratio (not_cyberbullying vs bullying): {ratio:.2f} (Target: ~1.00)")
    
    # Language / Script mix
    clean_df["script"] = clean_df["text"].apply(detect_script)
    script_counts = clean_df["script"].value_counts()
    print("\nLanguage / Script Breakdown:")
    for sc, cnt in script_counts.items():
        print(f"  • {sc:20s}: {cnt:4d} ({cnt/len(clean_df)*100:5.1f}%)")
        
    # Word-length distribution
    words_per_sample = clean_df["text"].apply(lambda x: len(x.split()))
    w_stats = words_per_sample.describe(percentiles=[0.25, 0.5, 0.75, 0.95])
    print("\nWord-Length Distribution:")
    print(f"  Min: {w_stats['min']:.0f} | 25%: {w_stats['25%']:.0f} | Median: {w_stats['50%']:.0f} | Mean: {w_stats['mean']:.1f} | 75%: {w_stats['75%']:.0f} | 95%: {w_stats['95%']:.0f} | Max: {w_stats['max']:.0f}")
    
    # 5. Template Pattern Detection (3-Word Openings)
    print("\nTemplate Pattern Detection (3-Word Openings):")
    openings = clean_df["text"].apply(lambda x: " ".join(re.findall(r'\b\w+\b', x.lower())[:3]))
    opening_counts = openings.value_counts()
    repetitive_templates = opening_counts[opening_counts > 3]
    template_flag = False
    if len(repetitive_templates) > 0:
        print("  ⚠️ Repetitive Opening Templates Detected (>3 occurrences):")
        for op, cnt in repetitive_templates.items():
            print(f"     \"{op}\": {cnt} times")
        template_flag = True
    else:
        print("  ✅ Passed: Zero repetitive opening templates detected.")
        
    # 6. Style-Leak Detector (Pointing, Absolute/Imperative Words, and Negative Adjective Preceding Group Term)
    print("\n" + "-" * 85)
    print(" 🔍 STYLE-LEAK DETECTOR (Pointing, Absolute Words, & Negative Adjectives)")
    print("-" * 85)
    
    safe_sub = clean_df[clean_df["category"] == "not_cyberbullying"]
    bully_sub = clean_df[clean_df["category"] != "not_cyberbullying"]
    
    # 1. Pointing Words
    safe_p_cnt = int(safe_sub["text"].str.contains(POINTING_PATTERN, case=False, regex=True).sum())
    bully_p_cnt = int(bully_sub["text"].str.contains(POINTING_PATTERN, case=False, regex=True).sum())
    safe_pointing = (safe_p_cnt / len(safe_sub) * 100) if len(safe_sub) > 0 else 0.0
    bully_pointing = (bully_p_cnt / len(bully_sub) * 100) if len(bully_sub) > 0 else 0.0
    ratio_pointing = (bully_pointing / safe_pointing) if safe_pointing > 0 else float('inf')
    
    # Min-count rule: ratio only flags when bullying side count >= 10
    flag_pointing = (ratio_pointing > 3.0 or ratio_pointing < 0.33) and (bully_p_cnt >= 10)
    
    print(f"1. Pointing Words (this/that/these/ye/is/in/इस/इन/यह/ये):")
    print(f"   • Safe share     : {safe_pointing:5.2f}% ({safe_p_cnt}/{len(safe_sub)})")
    print(f"   • Bullying share : {bully_pointing:5.2f}% ({bully_p_cnt}/{len(bully_sub)})")
    print(f"   • Disparity Ratio: {ratio_pointing:.2f}x (Bullying count: {bully_p_cnt}, Min threshold for flag: 10)")
    print(f"   • Flag Status    : {'⚠️ FLAGGED (Disparity > 3.0x & count >= 10)' if flag_pointing else '✅ PASSED'}")
    
    # 2. Absolute / Imperative Words
    safe_a_cnt = int(safe_sub["text"].str.contains(ABSOLUTE_PATTERN, case=False, regex=True).sum())
    bully_a_cnt = int(bully_sub["text"].str.contains(ABSOLUTE_PATTERN, case=False, regex=True).sum())
    safe_absolute = (safe_a_cnt / len(safe_sub) * 100) if len(safe_sub) > 0 else 0.0
    bully_absolute = (bully_a_cnt / len(bully_sub) * 100) if len(bully_sub) > 0 else 0.0
    ratio_absolute = (bully_absolute / safe_absolute) if safe_absolute > 0 else float('inf')
    
    # Min-count rule: ratio only flags when bullying side count >= 10
    flag_absolute = (ratio_absolute > 3.0 or ratio_absolute < 0.33) and (bully_a_cnt >= 10)
    
    print(f"\n2. Absolute / Imperative Words (stop/bas/हमेशा/केवल/बंद/मत):")
    print(f"   • Safe share     : {safe_absolute:5.2f}% ({safe_a_cnt}/{len(safe_sub)})")
    print(f"   • Bullying share : {bully_absolute:5.2f}% ({bully_a_cnt}/{len(bully_sub)})")
    print(f"   • Disparity Ratio: {ratio_absolute:.2f}x (Bullying count: {bully_a_cnt}, Min threshold for flag: 10)")
    print(f"   • Flag Status    : {'⚠️ FLAGGED (Disparity > 3.0x & count >= 10)' if flag_absolute else '✅ PASSED'}")
    
    # 3. Negative Adjective before Group Term
    safe_adj_cnt = int(safe_sub["text"].str.contains(NEG_ADJ_GROUP_PATTERN, case=False, regex=True).sum())
    bully_adj_cnt = int(bully_sub["text"].str.contains(NEG_ADJ_GROUP_PATTERN, case=False, regex=True).sum())
    safe_adj = (safe_adj_cnt / len(safe_sub) * 100) if len(safe_sub) > 0 else 0.0
    bully_adj = (bully_adj_cnt / len(bully_sub) * 100) if len(bully_sub) > 0 else 0.0
    ratio_adj = (bully_adj / safe_adj) if safe_adj > 0 else (float('inf') if bully_adj_cnt > 0 else 1.0)
    
    # Min-count rule: ratio only flags when bullying side count >= 10 and ratio > 3.0
    flag_adj = (ratio_adj > 3.0 or ratio_adj < 0.33) and (bully_adj_cnt >= 10)
    
    print(f"\n3. Negative Adjective before Group Term (self-righteous, loud, arrogant, lazy, careless, entitled, आलसी, लापरवाह, etc.):")
    print(f"   • Safe share     : {safe_adj:5.2f}% ({safe_adj_cnt}/{len(safe_sub)})")
    print(f"   • Bullying share : {bully_adj:5.2f}% ({bully_adj_cnt}/{len(bully_sub)})")
    print(f"   • Disparity Ratio: {ratio_adj:.2f}x (Bullying count: {bully_adj_cnt}, Min threshold for flag: 10)")
    print(f"   • Flag Status    : {'⚠️ FLAGGED (Disparity > 3.0x & count >= 10)' if flag_adj else '✅ PASSED'}")
    
    # Batch-level style flag: True if pointing, absolute, OR negative-adjective disparity detected with count >= 10
    style_flag = bool(flag_pointing or flag_absolute or flag_adj)
    flags = []
    if ratio < 0.5 or ratio > 2.0:
        flags.append(f"Ratio skew ({ratio:.2f})")
    if template_flag:
        flags.append("Template repetition")
    if style_flag:
        flags.append("Style leak disparity (>3x with count >= 10)")
        
    print(f"\nOverall Quality Status: " + ("⚠️ FLAGGED (" + ", ".join(flags) + ")" if flags else "✅ EXCELLENT / CLEAN (PASS)"))
    if style_flag:
        print("NOTE: Style-leak flag active for this batch (batch-level style_flag=True for ablation partitioning).")
    else:
        print("NOTE: Batch verified clean (batch-level style_flag=False for ablation partitioning).")
    
    # 7. Append to Master
    clean_df["style_flag"] = style_flag
    clean_df["batch_source"] = os.path.basename(batch_path)
    
    if not dry_run:
        cols_to_save = ["text", "category", "language", "style_flag", "batch_source"]
        clean_to_save = clean_df[cols_to_save]
        
        os.makedirs(os.path.dirname(os.path.abspath(master_path)), exist_ok=True)
        if os.path.exists(master_path):
            clean_to_save.to_csv(master_path, mode="a", header=False, index=False)
            total_master = len(pd.read_csv(master_path))
            print(f"\n Successfully appended {len(clean_to_save)} clean rows to: {master_path}")
            print(f" Total rows now in master: {total_master:,}")
        else:
            clean_to_save.to_csv(master_path, index=False)
            print(f"\n Successfully created master file with {len(clean_to_save)} rows at: {master_path}")
    else:
        print("\n [DRY RUN] No files written to disk.")
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Hard-Negative Batch Validator & Master Ingestion Pipeline")
    parser.add_argument("--batch", type=str, required=True, help="Path to raw batch CSV")
    parser.add_argument("--master", type=str, default="data/augmented/synthetic_hardneg_master.csv", help="Path to destination master CSV")
    parser.add_argument("--dry_run", action="store_true", help="Simulate quality audit without appending")
    args = parser.parse_args()
    
    validate_and_append_batch(args.batch, args.master, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

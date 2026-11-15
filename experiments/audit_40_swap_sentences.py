import os
import sys
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.abspath('.')

# 1. Load the 40 swap sentences
df_hn = pd.read_csv(os.path.join(ROOT_DIR, "data", "validation", "hardneg_synth_test_set.csv"))
safe_hn = df_hn[df_hn["category"] == "not_cyberbullying"].copy().reset_index(drop=True)

id_pattern = re.compile(
    r'(?<!\w)(mughal[s]?|मुग़ल|मुगल|muslim[s]?|musalman[s]?|मुस्लिम|मुसलमान|hindu[s]?|हिन्दू|हिंदू|sikh[s]?|सिख|christian[s]?|isaai|esai|ईसाई|bihari[s]?|बिहारी|madrasi[s]?|मद्रासी|women|woman|aurat|aurtein|महिला|महिलाएं|महिलाओ|औरत|औरतें|girl[s]?|ladki|ladkiyan|ladkiyon|लड़की|लड़कियां|लड़कियों|old\s+people|elderly|buddhe|budhe|बुड्ढे|बुजुर्ग|बूढ़े|बूढ़ा|boomer[s]?)(?!\w)',
    re.IGNORECASE
)

safe_hn["has_id"] = safe_hn["text"].fillna("").str.contains(id_pattern)
with_id_df = safe_hn[safe_hn["has_id"]].copy().reset_index(drop=True)
sample_40 = with_id_df.head(40).copy()

print(f"Total extracted swap sentences: {len(sample_40)}")
print(f"Source file: data/validation/hardneg_synth_test_set.csv (filtered: category=='not_cyberbullying', matches id_pattern, head(40))")
print(f"Languages in 40 sentences: {sample_40.get('language', pd.Series()).value_counts().to_dict()}")

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text

sample_texts = sample_40['text'].tolist()
sample_norms = [normalize_text(t) for t in sample_texts]

corpora = {
    'combined_train.parquet': ('data/processed/combined_train.parquet', 'parquet'),
    'synthetic_hardneg_master.csv': ('data/augmented/synthetic_hardneg_master.csv', 'csv'),
    'batch7_raw.csv': ('data/augmented/batch7_raw.csv', 'csv'),
    'batch11_raw.csv': ('data/augmented/batch11_raw.csv', 'csv'),
    'batch12_raw.csv': ('data/augmented/batch12_raw.csv', 'csv'),
    'batch13_raw.csv': ('data/augmented/batch13_raw.csv', 'csv'),
    'batch14_raw.csv': ('data/augmented/batch14_raw.csv', 'csv'),
    'civil_safe_identity.csv': ('data/augmented/civil_safe_identity.csv', 'csv'),
}

print("\n" + "=" * 95)
print("LEAKAGE AUDIT FOR 40 IDENTITY-SWAP SENTENCES AGAINST TRAINING / MASTER / BATCH CORPORA")
print("=" * 95)

all_pass = True

for cname, (cpath, ctype) in corpora.items():
    if not os.path.exists(cpath):
        print(f"[SKIP] {cname}: Not found")
        continue
    if ctype == 'parquet':
        cdf = pd.read_parquet(cpath)
    else:
        cdf = pd.read_csv(cpath)
    
    col = 'text' if 'text' in cdf.columns else ('cleaned_text' if 'cleaned_text' in cdf.columns else 'tweet_text')
    ctexts = cdf[col].dropna().astype(str).tolist()
    cnorms = [normalize_text(t) for t in ctexts]
    
    # Exact matches
    exact_set = set(ctexts)
    exact_matches = [i for i, t in enumerate(sample_texts) if t in exact_set]
    
    # Normalized matches
    norm_set = set(cnorms)
    norm_matches = [i for i, t in enumerate(sample_norms) if t in norm_set and i not in exact_matches]
    
    # Fuzzy char 3-5-gram TF-IDF cosine >= 0.80
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1)
    all_n = sample_norms + cnorms
    mat = vec.fit_transform(all_n)
    s_mat = mat[:len(sample_norms)]
    c_mat = mat[len(sample_norms):]
    sim = cosine_similarity(s_mat, c_mat)
    
    fuzzy_matches = []
    for i in range(len(sample_norms)):
        if i in exact_matches or i in norm_matches:
            continue
        max_s = float(np.max(sim[i]))
        if max_s >= 0.80:
            fuzzy_matches.append((i, max_s))
            
    total_leaks = len(exact_matches) + len(norm_matches) + len(fuzzy_matches)
    status = "PASS" if total_leaks == 0 else "FAIL"
    if total_leaks > 0:
        all_pass = False
    print(f"[{status}] {cname:<30} | Exact: {len(exact_matches):>2} | Norm: {len(norm_matches):>2} | Fuzzy (>=0.80): {len(fuzzy_matches):>2} | Total: {total_leaks:>2}")

print("\nOVERALL STATUS: " + ("ALL PASS (0 leaks across all corpora)" if all_pass else "LEAK DETECTED"))


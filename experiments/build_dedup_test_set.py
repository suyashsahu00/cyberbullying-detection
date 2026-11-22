import os, sys
ROOT_DIR = os.path.abspath('.')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.append_batch_to_master import normalize_text

train_df = pd.read_parquet('data/processed/combined_train.parquet')
test_df = pd.read_parquet('data/processed/combined_test.parquet')

train_col = 'cleaned_text' if 'cleaned_text' in train_df.columns else 'tweet_text'
test_col = 'cleaned_text' if 'cleaned_text' in test_df.columns else 'tweet_text'

train_df['norm'] = train_df[train_col].astype(str).apply(normalize_text)
test_df['norm'] = test_df[test_col].astype(str).apply(normalize_text)

train_raw_set = set(train_df[train_col].dropna().astype(str).str.strip())
train_norm_set = set(train_df['norm'])

# 1. Exact raw & normalized leaks
exact_indices = set()
for idx, row in test_df.iterrows():
    raw_val = str(row[test_col]).strip()
    norm_val = row['norm']
    if raw_val in train_raw_set or norm_val in train_norm_set:
        exact_indices.add(idx)

print(f"Exact raw/norm overlap in test set: {len(exact_indices)} rows")

# 2. Fuzzy char 3-5-gram TF-IDF cosine >= 0.80
print("Computing fuzzy similarity...")
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
vectorizer.fit(train_df['norm'].tolist() + test_df['norm'].tolist())

test_vecs = vectorizer.transform(test_df['norm'])
train_vecs = vectorizer.transform(train_df['norm'])

chunk_size = 500
fuzzy_ge_80_indices = set()

for i in range(0, len(test_df), chunk_size):
    chunk = test_vecs[i:i+chunk_size]
    sim = cosine_similarity(chunk, train_vecs)
    max_sim = np.max(sim, axis=1)
    for idx_in_chunk, s in enumerate(max_sim):
        if s >= 0.80:
            test_idx = i + idx_in_chunk
            fuzzy_ge_80_indices.add(test_idx)

print(f"Fuzzy cosine >= 0.80 overlap in test set: {len(fuzzy_ge_80_indices)} rows")

all_leaks_to_remove = exact_indices.union(fuzzy_ge_80_indices)
print(f"Total union of overlap test rows to remove: {len(all_leaks_to_remove)} rows ({len(all_leaks_to_remove)/len(test_df)*100:.2f}%)")

# Filter test set
dedup_test_df = test_df.drop(index=list(all_leaks_to_remove)).copy().reset_index(drop=True)
print(f"\nDeduplicated Test Set Shape: {dedup_test_df.shape}")

print("\n--- Original Blind Test Per-Class Distribution (5,242) ---")
print(test_df['cyberbullying_type'].value_counts())

print("\n--- Deduplicated Blind Test Per-Class Distribution ---")
print(dedup_test_df['cyberbullying_type'].value_counts())

# Save deduplicated test set
out_parquet = "data/processed/combined_test_dedup.parquet"
out_csv = "data/processed/combined_test_dedup.csv"
dedup_test_df.to_parquet(out_parquet, index=False)
dedup_test_df.to_csv(out_csv, index=False)
print(f"\nSaved deduplicated blind test set to:\n  - {out_parquet}\n  - {out_csv}")


import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def map_bully_target(row):
    """
    Standardizes BullyExplain targets to the 6-class taxonomy used by Kaggle & baseline model.
    """
    if row.get('Bully_Label') == 'Non_bully' or row.get('is_bully') == 0:
        return 'not_cyberbullying'
    
    target = str(row.get('Target', '')).strip()
    if target == 'Religion':
        return 'religion'
    elif target in ['Gender', 'Sexual Oriantation']:
        return 'gender'
    elif target in ['Race', 'Community']:
        return 'ethnicity'
    elif target == 'Age':
        return 'age'
    else:
        return 'other_cyberbullying'

def merge_and_save():
    print("=" * 70)
    print("MERGING KAGGLE (ENGLISH) & BULLYEXPLAIN (HINGLISH) DATASETS")
    print("=" * 70)

    # 1. Load Kaggle splits
    k_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "kaggle_train.parquet"))
    k_val = pd.read_parquet(os.path.join(PROCESSED_DIR, "kaggle_val.parquet"))
    k_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "kaggle_test.parquet"))

    # 2. Load BullyExplain splits
    b_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "bullyexplain_train.parquet"))
    b_val = pd.read_parquet(os.path.join(PROCESSED_DIR, "bullyexplain_val.parquet"))
    b_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "bullyexplain_test.parquet"))

    # Standardize columns for BullyExplain
    for df in [b_train, b_val, b_test]:
        df['cyberbullying_type'] = df.apply(map_bully_target, axis=1)
        df['tweet_text'] = df['tweet']
        if 'language' not in df.columns:
            df['language'] = 'Hinglish'

    columns_to_keep = ['tweet_text', 'cleaned_text', 'cyberbullying_type', 'language']

    # Combine splits
    comb_train = pd.concat([k_train[columns_to_keep], b_train[columns_to_keep]], ignore_index=True)
    comb_val = pd.concat([k_val[columns_to_keep], b_val[columns_to_keep]], ignore_index=True)
    comb_test = pd.concat([k_test[columns_to_keep], b_test[columns_to_keep]], ignore_index=True)

    # Shuffle datasets
    comb_train = comb_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    comb_val = comb_val.sample(frac=1.0, random_state=42).reset_index(drop=True)
    comb_test = comb_test.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Print exact shape and value_counts
    print(f"\n--- Combined Training Set ({len(comb_train):,} rows) ---")
    print(comb_train['cyberbullying_type'].value_counts())
    print("\nTraining Language Distribution:")
    print(comb_train['language'].value_counts())

    print(f"\n--- Combined Validation Set ({len(comb_val):,} rows) ---")
    print(comb_val['cyberbullying_type'].value_counts())

    print(f"\n--- Combined Test Set ({len(comb_test):,} rows) ---")
    print(comb_test['cyberbullying_type'].value_counts())

    # Save to disk
    train_parquet = os.path.join(PROCESSED_DIR, "combined_train.parquet")
    train_csv = os.path.join(PROCESSED_DIR, "combined_train.csv")
    val_parquet = os.path.join(PROCESSED_DIR, "combined_val.parquet")
    val_csv = os.path.join(PROCESSED_DIR, "combined_val.csv")
    test_parquet = os.path.join(PROCESSED_DIR, "combined_test.parquet")
    test_csv = os.path.join(PROCESSED_DIR, "combined_test.csv")

    comb_train.to_parquet(train_parquet, index=False)
    comb_train.to_csv(train_csv, index=False)
    comb_val.to_parquet(val_parquet, index=False)
    comb_val.to_csv(val_csv, index=False)
    comb_test.to_parquet(test_parquet, index=False)
    comb_test.to_csv(test_csv, index=False)

    print("\n Successfully saved combined dataset files to data/processed/:")
    print(f"  - {train_parquet}")
    print(f"  - {val_parquet}")
    print(f"  - {test_parquet}")

if __name__ == "__main__":
    merge_and_save()

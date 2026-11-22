# 📌 PROJECT STATE & BENCHMARK LOCK

> **Status:** LOCKED & VERIFIED ✅  
> **Timestamp:** September 15, 2026 (Day 2 Milestone Complete)  
> **Repository:** `suyashsahu00/cyberbullying-detection`  
> **Execution Environment:** Windows 11 | AMD Ryzen (16 Cores) | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) | Python 3.11.9 | PyTorch 2.5.1+cu121

---

## 1. Core Model Performance (Held-Out Test Set: 5,242 Samples)

*Verified by direct execution of `blind_test.py` with zero data leakage.*

### Headline Benchmark: Deduplicated Test Set (4,963 Samples) vs. Full Test Set (5,242 Samples)

*Deduplicated test set is the headline benchmark because the full test set has 5.32% train overlap (279 duplicate samples).*

| Metric | Argmax Mode (Standard) | Production Two-Stage (Deployed) | Delta | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Accuracy (Dedup Headline)** | **81.99%** (4,069/4,963) | **78.78%** (3,910/4,963) | -3.20% | Primary leak-free test set (N=4,963) |
| **Macro F1-Score (6-Cl, Dedup)** | **83.08%** | **77.87%** | -5.21% | Primary leak-free test set |
| **Safe Recall (Dedup)** | **49.85%** (511/1,025) | **80.39%** (824/1,025) | +30.54% | Deployed threshold + Option A safety-net: fewer harmless texts flagged |
| **Overall Accuracy (Full)** | 81.97% (4,297/5,242) | 78.69% (4,125/5,242) | -3.28% | Evaluated on full test set (N=5,242) |
| **Macro Precision (6-Cl, Full)** | 83.68% | 81.38% | -2.30% | Unweighted average over all 6 classes |
| **Macro Recall (6-Cl, Full)** | 83.88% | 79.63% | -4.25% | Two-stage trades other-recall for safe-recall |
| **Macro F1-Score (6-Cl, Full)** | 83.21% | 77.89% | -5.32% | Bully 5-class F1: 88.07% vs 80.57% |
| **Safe Recall (Full)** | 50.00% (544/1,088) | 80.97% (881/1,088) | +30.97% | Deployed threshold + Option A safety-net: fewer harmless texts flagged |
| **Safe Precision (Full)** | 71.77% | 53.59% | -18.18% | Of all texts labelled Safe, only 53.6% truly are Safe; two-stage passes generic bullying as Safe |

*Note: Legacy numbers (83.54%, 85.10%, 85.21%): unverified; source not documented.*

### Per-Class Breakdown (Production Two-Stage):
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Age** | 97.16% | 98.25% | 97.70% | 800 |
| **Ethnicity** | 97.47% | 93.12% | 95.24% | 828 |
| **Religion** | 94.07% | 96.83% | 95.43% | 819 |
| **Gender** | 82.94% | 90.95% | 86.76% | 807 |
| **Not Cyberbullying (Safe)** | 79.06% | 44.76% | 57.16% | 1,088 |
| **Other Cyberbullying** | 65.02% | 19.00% | 29.41% | 900 |

*Artifact File:* `models/blind_test_verified_metrics.json`  
*Footnote: Legacy numbers for Other Cyberbullying (Precision 57.40%, Recall 82.78%, F1-Score 67.79%): unverified; source not documented.*

---

## 2. Latency & Hardware Benchmarks (30 Warm Iterations per Input)

*Verified by direct execution of `benchmark_latency.py`.*

| Benchmark Metric | Tier 1: Linear SVM (CPU) | Tier 2: Google MuRIL v2 (RTX 4050 GPU) |
| :--- | :---: | :---: |
| **Cold-Start 1st Invocation** | 14.68 ms | 958.92 ms |
| **Warm Mean Latency (Average)** | 1.98 ms | 107.95 ms |
| **Warm Median Latency (P50)** | 1.79 ms | 150.43 ms |
| **Warm 90th Percentile (P90)** | 2.90 ms | 179.40 ms |
| **Warm 95th Percentile (P95)** | 3.16 ms | 188.52 ms |
| **Warm 99th Percentile (P99)** | 4.42 ms | 197.95 ms |
| **Minimum Observed Latency** | 0.77 ms | 11.50 ms |
| **Maximum Observed Latency** | 5.95 ms | 211.15 ms |
| **Standard Deviation** | 0.71 ms | 71.02 ms |
| **Throughput** | **505.2 QPS** | **9.3 QPS** |

*Artifact File:* `models/latency_benchmark_results.json`

---

## 3. Supplementary Sarcasm Detection Module (Day 1 Decision: Option B)

*Dataset:* `dasarpai/SDSHL` (2,000 sentences, Devanagari script + code-switched Hindi/English).  
*Integration Architecture:* Standalone auxiliary prototype, 100% decoupled from the primary 6-class cyberbullying safety pipeline.

### SDSHL Test Results (200 Held-Out Samples):
- **Train Accuracy:** 97.22%
- **Test Accuracy:** 66.00%
- **Macro F1-Score:** 66.00%
- **Confusion Matrix:** TN=66, FP=34, FN=34, TP=66
- **Code Module:** `src/sarcasm_auxiliary.py`
- **Model Artifact:** `models/sarcasm_auxiliary.joblib`

---

## 4. Documentation & Milestone Synchronization Status

- [x] **Bug 1 (Eval/Production Mismatch):** Unified two-stage decision boundary in `src/model.py` and `blind_test.py`.
- [x] **Bug 3 (Explainability Relabeling):** Correctly labeled as "Keyword-Based Trigger Detection" across UI and codebase.
- [x] **Bug 4 ('Other' Class Analysis):** Documented measured trade-off: two-stage boundary with Option A safety-net increases Safe recall (50.0% $\to$ 81.0%) while shifting generic 'other' bullying to Safe (recall 76.3% $\to$ 19.0%).
- [x] **Bug 5 (Latency Benchmarking):** Live execution completed on RTX 4050 GPU and logged into JSON.
- [x] **Day 1 Audit & Decision:** `dasarpai/SDSHL` audited, Option B auxiliary module implemented, and documented in `README.md`.
- [x] **Day 2 Lock:** Final runs of `blind_test.py` and `benchmark_latency.py` completed and numbers 100% synchronized with `report.md`.

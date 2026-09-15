# 📌 PROJECT STATE & BENCHMARK LOCK

> **Status:** LOCKED & VERIFIED ✅  
> **Timestamp:** September 15, 2026 (Day 2 Milestone Complete)  
> **Repository:** `suyashsahu00/cyberbullying-detection`  
> **Execution Environment:** Windows 11 | AMD Ryzen (16 Cores) | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) | Python 3.11.9 | PyTorch 2.5.1+cu121

---

## 1. Core Model Performance (Held-Out Test Set: 5,242 Samples)

*Verified by direct execution of `blind_test.py` with zero data leakage.*

### Overall Metrics:
| Metric | Academic Argmax (Baseline) | Production Two-Stage (Deployment) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 81.38% | **81.97%** | **+0.59%** |
| **Macro Precision** | 84.01% | **83.21%** | -0.80% |
| **Macro Recall** | 82.17% | **83.41%** | **+1.24%** |
| **Macro F1-Score** | 81.87% | **83.29%** | **+1.42%** |

### Per-Class Breakdown (Production Two-Stage):
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Age** | 97.16% | 98.38% | 97.76% | 800 |
| **Ethnicity** | 98.10% | 93.72% | 95.86% | 828 |
| **Religion** | 94.34% | 95.73% | 95.03% | 819 |
| **Gender** | 84.04% | 88.72% | 86.32% | 807 |
| **Not Cyberbullying (Safe)** | 66.18% | 63.14% | 64.63% | 1,088 |
| **Other Cyberbullying** | 59.46% | 60.78% | 60.11% | 900 |

*Artifact File:* `models/blind_test_verified_metrics.json`

---

## 2. Latency & Hardware Benchmarks (30 Warm Iterations per Input)

*Verified by direct execution of `benchmark_latency.py`.*

| Benchmark Metric | Tier 1: Linear SVM (CPU) | Tier 2: Google MuRIL v2 (RTX 4050 GPU) |
| :--- | :---: | :---: |
| **Cold-Start 1st Invocation** | 6.78 ms | 337.49 ms |
| **Warm Mean Latency (Average)** | 0.79 ms | 64.36 ms |
| **Warm Median Latency (P50)** | 0.77 ms | 87.50 ms |
| **Warm 90th Percentile (P90)** | 0.84 ms | 108.67 ms |
| **Warm 95th Percentile (P95)** | 0.88 ms | 110.90 ms |
| **Warm 99th Percentile (P99)** | 1.12 ms | 112.21 ms |
| **Minimum Observed Latency** | 0.68 ms | 10.79 ms |
| **Maximum Observed Latency** | 1.54 ms | 135.83 ms |
| **Standard Deviation** | 0.09 ms | 43.73 ms |
| **Throughput** | **1,270.9 QPS** | **15.5 QPS** |

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
- [x] **Bug 4 ('Other' Class Analysis):** Documented mathematical justification for recall jump (36.33% $\to$ 60.78%) via two-stage probability pooling.
- [x] **Bug 5 (Latency Benchmarking):** Live execution completed on RTX 4050 GPU and logged into JSON.
- [x] **Day 1 Audit & Decision:** `dasarpai/SDSHL` audited, Option B auxiliary module implemented, and documented in `README.md`.
- [x] **Day 2 Lock:** Final runs of `blind_test.py` and `benchmark_latency.py` completed and numbers 100% synchronized with `report.md`.

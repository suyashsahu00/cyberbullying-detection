# 🎤 CyberGuard AI: Presentation & Live Demo Master Guide

> **Target Presentation:** PPT-2 / Viva (September 21, 2026)  
> **Project:** Multilingual Cyberbullying Detection & Explainability (CyberGuard AI)  
> **Repository:** `suyashsahu00/cyberbullying-detection`

---

## 🎬 1. Refreshed 3-Minute Live Demo Flow (Zero Boring Examples)

All previous repetitive examples have been replaced with **fresh, real-world test cases** from our newly integrated **SDSHL dataset**, **BullyExplain Hinglish benchmark**, and **Kaggle test set**:

```
[0:00 - 0:45] Fresh Safe Baseline ➔ [0:45 - 1:30] Real Gender Attack ➔ [1:30 - 2:15] Code-Switched Hinglish Slur ➔ [2:15 - 3:00] SDSHL Sarcasm Nuance
```

---

### Step 1: Fast Non-Abusive Filtering (Tier 1: Linear SVM)
* **Goal:** Demonstrate sub-millisecond safe-comment filtering without wasting GPU compute.
* **Input Text (Fresh Sample):**
  > `"Hope you have a wonderful day ahead with your family and loved ones!"`
* **What to Click:** Click the **"Friendly"** preset pill $\rightarrow$ notice text populates into the box $\rightarrow$ Click **"Analyze Comment"**.
* **Expected Result:**  
  - Verdict: 🟢 **Not Cyberbullying** (Safe)
  - Latency: **~0.8 ms**
  - Language: `English`
* **What to Say to the Panel:**
  > *"Sir, social media par 90%+ comments non-abusive hote hain. Hamara Tier 1 Classical Baseline (Linear SVM) CPU par sirf ~0.8 ms me in safe queries ko filter kar deta hai, giving ~1,270 Queries/Sec throughput."*

---

### Step 2: Contextual Gender Harassment (Tier 2: Google MuRIL)
* **Goal:** Show identity-targeted harassment detection and explainability trigger words.
* **Input Text (Fresh Sample):**
  > `"Women are too emotional to lead, stay home where you belong bitch."`
* **What to Click:** Click **"Gender Attack"** preset pill $\rightarrow$ Click **"Analyze Comment"**.
* **Expected Result:**  
  - Verdict: 🔴 **Cyberbullying Detected**
  - Category: **Gender** (High Confidence $>90\%$)
  - Triggers Highlighted: `<mark>bitch</mark>`, `<mark>women</mark>`
* **What to Say to the Panel:**
  > *"Jab comment identity-targeted ho, hamara deep model identity category tag (Gender) detect karta hai aur transparent explainability ke zariye trigger words highlight kar deta hai."*

---

### Step 3: Code-Switched Hinglish Slur Attack (BullyExplain Real Sample)
* **Goal:** Show why standard English-only BERT models fail and MuRIL succeeds.
* **Input Text (Fresh Test Sample from BullyExplain):**
  > `"Chuthiya tera dharm apne pas rakh aur bakwaas mat kar saale."`
* **What to Click:** Click **"Hinglish Bullying"** preset pill $\rightarrow$ Click **"Analyze Comment"**.
* **Expected Result:**  
  - Verdict: 🔴 **Cyberbullying Detected**
  - Language Detected: `Hinglish`
  - Category: **Religion / Other Cyberbullying**
  - Abusive Triggers: `<mark>Chuthiya</mark>`, `<mark>bakwaas</mark>`, `<mark>saale</mark>`
* **What to Say to the Panel:**
  > *"Monolingual English models is code-mixed sentence ko parse nahi kar paate. Google MuRIL ka 17-Indian-language subword tokenizer Hinglish slurs aur colloquial syntax ko naturally understand karta hai."*

---

### Step 4: Sarcasm Nuance Test (From Newly Audited SDSHL Dataset)
* **Goal:** Showcase the sarcasm auxiliary prototype and architectural decoupling.
* **Input Text (Fresh Sample from dasarpai/SDSHL):**
  > `"अंग्रेजी नहीं आती है इसलिए हिन्दी ट्विट ज्यादा करते हैं क्या बात है!"`
* **What to Click:** Click **"SDSHL Sarcasm"** preset pill $\rightarrow$ Click **"Analyze Comment"**.
* **Expected Result:**  
  - Language: `Hindi` (Devanagari Unicode detected)
  - Standalone Sarcasm Prediction (`src/sarcasm_auxiliary.py`): **Sarcastic (69.97% Confidence)**
* **What to Say to the Panel:**
  > *"Ye sentence surface level par compliment lag sakta hai ('क्या बात है!'), lekin sarcasm benchmark (SDSHL) par hamara auxiliary prototype ise 69.97% confidence ke saath Sarcastic flag karta hai. Humne ise primary pipeline se decoupled rakha hai taaki core safety boundaries dilute na hon."*

---

## 🎯 2. Tough Viva / Reviewer Questions & Bulletproof Answers

### Q1: *"Aapka 'Other Cyberbullying' recall 36.33% se 60.78% par kaise jump kar gaya? Kya koi data leakage thi?"*
> **Answer:**  
> *"No Sir, zero data leakage hai aur training weights bilkul unchanged hain. Ye naive `argmax` ka **probability fragmentation flaw** tha. Multiclass setting me agar safe probability 40% thi aur harassment probability 60% thi—lekin wo 60% paanch bullying classes me bat gayi (e.g. 25% other, 15% ethnicity, 10% age, 10% gender)—toh simple argmax 40% Safe ko winner ghoshit kar deta tha. Humne **Two-Stage Production Boundary** lagayi: pehle binary check hota hai ($1 - P(\text{Safe}) \ge 50\%$). Is single architectural fix ne noisy Other class ka recall naturally 36.33% se 60.78% par restore kar diya."*

### Q2: *"Sarcasm detection ko core model ke andar integrate kyu nahi kiya?"*
> **Answer:**  
> *"Sir, dasarpai/SDSHL jaise available Hinglish sarcasm datasets small (2,000 samples) hain aur primarily Devanagari script me hain. Unhe 50k-sample production cyberbullying model me forcefully merge karne se false-positive rate badh jata. Isliye humne **microservice-style architectural decoupling** chuni: core safety alag aur sarcasm standalone prototype (`src/sarcasm_auxiliary.py`) me independent."*

### Q3: *"64 ms vs 0.8 ms latency: do alag models kyu deploy kiye?"*
> **Answer:**  
> *"Ye hamara **Tiered Production Architecture** hai. Linear SVM 0.79 ms aur 1,270 QPS par CPU par chal sakta hai, jo high-volume streaming platforms ke liye ideal hai. MuRIL v2 GPU par 64.36 ms leta hai for deep contextual understanding. Live system dono modes seamlessly offer karta hai."*

### Q4: *"Explainability ke liye SHAP kyu nahi chala rahe live UI me?"*
> **Answer:**  
> *"SHAP perturbation-based method hai jo per-sentence 500+ passes leta hai, jisse latency 5–10 seconds ho jati hai—jo real-time chat moderation me completely unusable hai. Humne model-aligned Keyword-Based Trigger Detection aur token attribution use kiya hai jo $<2\text{ ms}$ me instant visual feedback render karta hai."*

---

## 📑 3. PPT-2 (21 Sept) Slide-by-Slide Blueprint

| Slide # | Title | Key Talking Points & Visuals |
| :---: | :--- | :--- |
| **1** | **Title Slide** | CyberGuard AI: Multilingual Cyberbullying Detection & Explainability<br>*Presenter: Suyash Sahu \| Supervisor: [Guide Name]* |
| **2** | **The Research Problem** | 1. Rise of online abuse in Indian social media.<br>2. Failure of monolingual models on Romanized Hinglish.<br>3. The subtlety of sarcastic harassment. |
| **3** | **Dataset Taxonomy** | Table showing 3 benchmarks:<br>• Kaggle (47.7k English)<br>• BullyExplain (6.4k Hinglish with rationales)<br>• SDSHL (2.0k Hinglish sarcasm pilot) |
| **4** | **System Architecture** | Dual-Tier Pipeline Diagram:<br>Input $\rightarrow$ Preprocessing & Language Tagger $\rightarrow$ Tier 1 SVM vs Tier 2 MuRIL $\rightarrow$ Explainability $\rightarrow$ UI. |
| **5** | **Language Detection Subsystem** | Unicode regex scan (`Hindi`) $\rightarrow$ Hinglish token ratio ($\ge 10\%$) $\rightarrow$ English default fallback. |
| **6** | **Decision Boundary Innovation (Bug 1)** | Argmax vs Production Two-Stage table.<br>Highlighting the **+24.45% recall jump** on the challenging `Other` class without retraining. |
| **7** | **Final Verified Performance (Offline Test)** | • Full Test Set: 5,242 samples (zero data leakage)<br>• **Overall Accuracy: 81.97%** \| **Macro F1: 83.29%**<br>• Per-class F1: Age (97.8%), Ethnicity (95.9%), Religion (95.0%), Gender (86.3%), Other (60.1%)<br>• GPU Batched Evaluation Throughput: **358.4 samples/sec (2.79 ms/sample)** on RTX 4050 |
| **8** | **Hardware & Latency Benchmark (Single-Query)** | Clear Hardware Disaggregation:<br>• **Tier 1 (Linear SVM on CPU)**: **0.79 ms Warm Mean** (0.77 ms P50, 1.12 ms P99) \| **1,270.9 QPS**<br>• **Tier 2 (Google MuRIL v2 on RTX 4050 GPU)**: **64.36 ms Warm Mean** (87.50 ms P50, 110.90 ms P95) \| **15.5 QPS**<br>• Cold-Start: SVM (6.78 ms) vs MuRIL GPU (337.49 ms) |
| **9** | **Auxiliary Sarcasm Pilot Study** | SDSHL dataset evaluation (66% F1). Rationale for maintaining architectural decoupling rather than polluting the safety boundary. |
| **10** | **Live Web Application & API** | UI Screenshots, real-time trigger badge highlighting, and Swagger/REST API format. |
| **11** | **Conclusion & Future Directions** | Production-ready real-time moderation tool; future expansion to acoustic tone analysis and Romanized Hinglish sarcasm corpora. |

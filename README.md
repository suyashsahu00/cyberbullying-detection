# CyberGuard AI: Multilingual Cyberbullying Detection & Explainability 🛡️

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20PyTorch-orange.svg)](https://flask.palletsprojects.com/)
[![Transformer](https://img.shields.io/badge/Model-Google%20MuRIL%20%7C%20Linear%20SVM-green.svg)](https://huggingface.co/google/muril-base-cased)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end NLP-powered web application and REST API for detecting cyberbullying and online harassment in social media comments and tweets. Supports multilingual text (**English**, **Hinglish**, and **Hindi**), categorizes target demographics (**Age**, **Gender**, **Ethnicity**, **Religion**, **Other**), and provides token-level explainability by highlighting trigger words and rationales.

---

## 🌟 Key Features

- 🎯 **Real-Time Detection Verdict**: Instant classification badge — `Cyberbullying Detected` (Red) vs `Not Cyberbullying` (Green).
- 🏷️ **Multi-Class Identity Categorization**: Identifies specific harassment types (`Age`, `Gender`, `Ethnicity`, `Religion`, `Other / Hinglish Slur`).
- 📊 **Confidence & Probability Distribution**: Animated confidence meter and probability bars across all target classes.
- 💡 **Token-Level Explainability**: Highlights and attributes exact abusive trigger words and harassment spans in red.
- 🌐 **Automated Language Detection**: Intelligent detection distinguishing pure **English**, code-switched **Hinglish** (Romanized Hindi), and **Hindi** (Devanagari).
- ⚡ **Dual AI Architecture**:
  - **Tier 1 (Fast Baseline)**: TF-IDF + Linear SVM (~0.6 ms inference on CPU).
  - **Tier 2 (Deep Transformer)**: Fine-tuned Google MuRIL for code-switched Hinglish and multilingual context (~20 ms on GPU).
- 🎨 **Modern Responsive UI**: Clean interface built with Bootstrap 5, FontAwesome, Google Fonts (*Outfit* & *Inter*), real-time character counters, and 1-click test presets.

---

## 🌐 How Language Detection (English vs Hinglish vs Hindi) Works

The system automatically identifies whether an incoming comment is in **English**, **Hinglish** (Hindi written in Latin script), or **Hindi** (Devanagari script) through an intelligent multi-stage pipeline located in `src/preprocessing.py`:

```
                          Input Comment / Tweet
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
1. Devanagari Unicode Check                             2. Latin Script Text
   ( [\u0900-\u097F] )                                          │
        │                                                       ▼
  Match found:                                    Tokenize into Words [a-z]+
   ➔ "Hindi"                                                    │
                                                                ▼
                                                   Compare with HINGLISH_KEYWORDS
                                                   (bhai, yaar, tera, kya, hai, saale,
                                                    chutiya, pagal, kuttiya, bakwas, etc.)
                                                                │
                                                                ▼
                                                   Calculate Hinglish Word Ratio
                                                                │
                                        ┌───────────────────────┴───────────────────────┐
                                        ▼                                               ▼
                         Ratio ≥ 10% OR Hinglish Count ≥ 2:                 Otherwise (Standard Vocab):
                                 ➔ "Hinglish"                                      ➔ "English"
```

### Detection Logic & Rules Breakdown:

| Priority | Language Target | Detection Mechanism | Example Input | Tagged Output |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Highest)** | **Hindi** | Unicode regex scan `[\u0900-\u097F]` for Devanagari characters | `"आप बहुत बुरे इंसान हैं"` | `Hindi` |
| **2** | **Hinglish** | Word tokenization + comparison against `HINGLISH_KEYWORDS` lexicon. Tagged if $\ge 10\%$ of words or $\ge 2$ marker keywords match | `"Bhai tu bilkul pagal aur kuttiya hai"` | `Hinglish` |
| **3 (Default)**| **English** | Fallback when neither Devanagari nor Hinglish keyword thresholds are met | `"Stop harassing people online you loser"` | `English` |

### Hinglish Marker Lexicon (`src/preprocessing.py`):
Includes conversational markers, pronouns, auxiliary verbs, and colloquial terms:
> `bhai`, `yaar`, `tera`, `teri`, `mera`, `meri`, `kya`, `hai`, `hain`, `karo`, `karna`, `raha`, `rahi`, `nahi`, `nhi`, `mat`, `bol`, `kaise`, `aur`, `sab`, `saale`, `kamina`, `pagal`, `chutiya`, `kuttiya`, `bakwas`, `gandu`, `harami`, etc.

---

## 🔄 Complete End-to-End System Working Process

The following diagram illustrates how an input text traverses the entire architecture from submission to final UI visualization:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. Frontend Web UI                              │
│   - User enters tweet/comment (or picks a 1-click Preset)              │
│   - User selects AI Model: Google MuRIL vs Classical Baseline          │
│   - AJAX POST Request sent to `/api/analyze`                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               2. Text Preprocessing (`src/preprocessing.py`)           │
│   - HTML entity decoding (`&amp;` ➔ `&`, `&lt;` ➔ `<`)                 │
│   - URL normalization (`https://...` ➔ `<URL>`)                        │
│   - Mentions normalization (`@username` ➔ `<USER>`)                    │
│   - Control characters stripped & whitespace normalized                │
│   - Text statistics extracted (Caps ratio, char count, word count)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             3. Language Detection (`src/preprocessing.py`)             │
│   - Devanagari scan ➔ Hindi                                            │
│   - Hinglish token density evaluation ➔ Hinglish                       │
│   - Standard English dictionary fallback ➔ English                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             4. Model Inference Engine (`src/model.py`)                 │
│                                                                        │
│   [Choice A: Google MuRIL Transformer]                                │
│   - Tokenized with MuRIL AutoTokenizer (max_length=128, padded)        │
│   - PyTorch forward pass (CUDA GPU or CPU)                             │
│   - Softmax probabilities computed: Safe vs Cyberbullying              │
│                                                                        │
│   [Choice B: Classical Baseline (Linear SVM)]                         │
│   - TF-IDF word & char n-grams extraction                              │
│   - 6-Class prediction: Age, Gender, Ethnicity, Religion, Other, Safe │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            5. Explainability Engine (`src/explainability.py`)          │
│   - Target lexicon lookup across identity categories                   │
│   - Word/phrase boundary matching & span extraction                    │
│   - Overlap resolution & XSS-safe HTML `<mark>` span generation        │
│   - Calculates individual trigger token attribution weights            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    6. REST API Response Packaging                      │
│   - Response formatted as JSON containing verdict, confidence,         │
│     category, highlighted HTML, latency in ms, and stats               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               7. Dynamic Frontend UI Rendering (`app.js`)              │
│   - Verdict Badge updated (Green Safe vs Red Bullying)                 │
│   - Animated confidence bar & probability breakdown rendered           │
│   - Highlighted HTML injected with red badges on trigger words         │
│   - Execution latency displayed in milliseconds                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Explainability & Target Categories

The explainability engine (`src/explainability.py`) matches and flags abusive spans across multi-class target categories:

| Target Category | Monitored Trigger Patterns & Keywords |
| :--- | :--- |
| **Age** | `boomer`, `old hag`, `senile`, `wrinkly`, `toddler`, `kiddo`, `dinosaur`, `fossil`, etc. |
| **Gender** | `bitch`, `whore`, `slut`, `cunt`, `kitchen`, `make me a sandwich`, `femcel`, `incel`, `hoe`, etc. |
| **Ethnicity** | Racial slurs, xenophobic attacks, `go back to your country`, `curryboy`, `illegal immigrant`, etc. |
| **Religion** | `terrorist`, `kafir`, `infidel`, `jihadist`, `extremist`, `ricebag`, `heathen`, etc. |
| **Other / Slurs** | `loser`, `ugly`, `kill yourself`, `stupid`, `idiot`, `chutiya`, `saale`, `kamina`, `harami`, `bakwas`, `pagal`, `kuttiya`, `gandu`, `kamine`, `madarchod`, `bhosdike`, `randi`, `kutte`, etc. |

---

## 🔤 Typo, Misspelling & Obfuscation Handling

A common cyberbullying evasion tactic is deliberately misspelling or elongating letters in offensive words (e.g. `stuuuupid`, `biiiitch`, `chuuutiya`, `looserrrr`, `pagallll`). The system captures these through three complementary layers:

1. **Subword Tokenization (Google MuRIL Transformer)**:
   - MuRIL breaks words into subword roots using **WordPiece** tokenization.
   - For example, `stuuupid` is tokenized into `['st', '##uuu', '##pid']`. Self-attention mechanisms combine these sub-tokens with contextual clues from surrounding words to preserve semantic intent and flag the comment.
2. **Flexible Character-Elongation Regex (`src/explainability.py`)**:
   - The trigger highlighter transforms each dictionary word into a repeated-character pattern:
     ```python
     # 'bitch' matches 'biiiitch', 'biiitch', 'bitchhhh'
     pattern = r'\b' + ''.join(c + '+' for c in word) + r'\b'
     ```
   - This ensures that even stretched or repeated character slurs are highlighted in the UI.
3. **Character n-Grams (Linear SVM Baseline)**:
   - The TF-IDF vectorizer captures sub-word character sequences ($n=3$ to $5$), making the baseline resilient to minor spelling variations.

---

## 📁 Project Structure

```
cyberbullying-detection/
├── backend/
│   ├── app.py                 # Flask server (API routes & UI template router)
│   └── requirements.txt       # Production dependencies
├── frontend/
│   ├── templates/
│   │   └── index.html         # Main Web application template (Jinja2 / HTML5)
│   └── static/
│       ├── css/
│       │   └── style.css      # Design tokens, gradients, badges & layout styles
│       └── js/
│           └── app.js         # Interactive DOM handling, API fetch & rendering
├── src/
│   ├── model.py               # Unified classifier (MuRIL & Linear SVM pipelines)
│   ├── preprocessing.py      # Multilingual text cleaning & language detection
│   └── explainability.py      # Trigger word lexicon & token-level HTML highlighter
├── models/
│   ├── baseline_model.pkl     # Pre-trained TF-IDF + Linear SVM 6-class classifier
│   └── muril_cyberbullying/   # Fine-tuned Google MuRIL model weights & tokenizer
├── data/
│   ├── raw/                   # Raw datasets (e.g. Cyberbullying & BullyExplain)
│   └── processed/             # Cleaned, balanced, and tokenized train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis & class distributions
│   ├── 02_preprocessing.ipynb # Text cleaning & language tagging pipelines
│   ├── 03_baseline_model.ipynb# Baseline multi-class training & evaluation
│   ├── 04_muril_finetune.ipynb# MuRIL transformer fine-tuning
│   └── 05_evaluation_shap.ipynb# Token rationale recall & SHAP attribution
├── pyproject.toml             # uv & project configuration
├── Procfile                   # Cloud deployment entry point (Gunicorn)
└── README.md
```

---

## 🚀 How to Start the Frontend & Application

The web frontend is integrated with the Flask backend. Starting the server will serve both the **Web User Interface** and the **REST API endpoints**.

### Prerequisites
- Python 3.9 or higher installed
- (Optional) NVIDIA GPU with CUDA for accelerated MuRIL transformer inference

---

### Step 1: Open the Project Directory

```bash
cd cyberbullying-detection
```

---

### Step 2: Set Up Environment & Install Dependencies

#### Option A: Using `uv` (Recommended — Ultra Fast)
```bash
# Sync dependencies automatically
uv sync
```

#### Option B: Using standard Python `venv` & `pip`

**On Windows (PowerShell):**
```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\Activate.ps1

# 3. Install required packages
pip install -r backend/requirements.txt
```

**On macOS / Linux:**
```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install required packages
pip install -r backend/requirements.txt
```

---

### Step 3: Run the Application Server

#### Using `uv`:
```bash
uv run python backend/app.py
```

#### Using Standard Python:
```bash
python backend/app.py
```

---

### Step 4: Open the Frontend in Your Browser

Once the server is running, open your web browser and navigate to:

👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** (or `http://localhost:5000`)

---

## 🖥️ Using the Web Interface

1. **AI Model Engine Selector**:
   - Choose between **Google MuRIL Transformer** (optimal for Hinglish and deep context) or **Classical Baseline** (Linear SVM for fast 6-class demographic categorization).
2. **Input Text Area**: Type or paste any tweet or comment (up to 500 characters).
3. **Quick Test Presets**: Click any preset pill (*Safe / Friendly*, *Hinglish Bullying*, *Gender Attack*, *Age Bullying*, *Ethnicity Attack*) to auto-fill and analyze.
4. **Keyboard Shortcut**: Press `Ctrl + Enter` (or `Cmd + Enter` on macOS) to instantly submit the text.
5. **Inspect the Moderation Report**:
   - **Verdict Badge**: Cyberbullying Detected vs Not Cyberbullying.
   - **Identified Category**: Demographic category tag (Age, Gender, Ethnicity, Religion, Other).
   - **Inference Latency**: Real-time server execution time in milliseconds.
   - **Language Tag**: Automatic indicator (`English`, `Hinglish`, `Hindi`).
   - **Keyword-based Trigger Detection**: Red-flagged trigger words extracted from the text.
   - **Class Probabilities**: Breakdown of confidence across classes.

---

## 📡 REST API Reference

The backend exposes a JSON REST API for integration into other applications or bots.

### 1. Health Check
`GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "Cyberbullying Detection & Explainability API",
  "version": "2.0.0",
  "models_available": ["Google MuRIL Transformer", "Linear SVM Baseline"]
}
```

---

### 2. Analyze Text
`POST /api/analyze`

**Request Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "text": "Go back to the kitchen and make me a sandwich bitch.",
  "model_choice": "muril"
}
```
> `model_choice` is optional and accepts `"muril"` (default) or `"baseline"`.

**Success Response (`200 OK`):**
```json
{
  "verdict": "Cyberbullying Detected",
  "is_cyberbullying": true,
  "category": "Gender",
  "confidence": 97.4,
  "language": "English",
  "latency_ms": 18.4,
  "model_used": "Google MuRIL Transformer (Multilingual/Hinglish)",
  "original_text": "Go back to the kitchen and make me a sandwich bitch.",
  "cleaned_text": "Go back to the kitchen and make me a sandwich bitch.",
  "explainability": {
    "trigger_words": ["bitch", "kitchen"],
    "highlighted_text": "Go back to the <mark class=\"trigger-highlight\" data-category=\"Gender\" title=\"Trigger: Gender\">kitchen</mark> and make me a sandwich <mark class=\"trigger-highlight\" data-category=\"Gender\" title=\"Trigger: Gender\">bitch</mark>.",
    "spans": [
      {
        "category": "Gender",
        "start": 15,
        "end": 22,
        "weight": 0.84,
        "word": "kitchen"
      },
      {
        "category": "Gender",
        "start": 47,
        "end": 52,
        "weight": 0.77,
        "word": "bitch"
      }
    ]
  },
  "all_probabilities": {
    "Safe / Non-Bully": 2.6,
    "Cyberbullying / Harassment": 97.4
  },
  "stats": {
    "caps_ratio": 0.0192,
    "char_count": 52,
    "language": "English",
    "word_count": 10
  }
}
```

---

## 🔬 Model Training & Jupyter Notebooks

To retrain the models or reproduce the experimental results, launch Jupyter Notebooks:

```bash
# With uv
uv run jupyter notebook

# With standard Python environment
jupyter notebook
```

Execute the notebooks in sequence:
1. `notebooks/01_eda.ipynb`: Dataset exploration and class distributions.
2. `notebooks/02_preprocessing.ipynb`: Text cleaning, tokenization, and language tagging.
3. `notebooks/03_baseline_model.ipynb`: TF-IDF + Linear SVM multi-class training and evaluation.
4. `notebooks/04_muril_finetune.ipynb`: Google MuRIL fine-tuning on multilingual/Hinglish dataset.
5. `notebooks/05_evaluation_shap.ipynb`: SHAP explainability and word rationale recall evaluation.

---

## 🛡️ Production Deployment

### Running with Gunicorn (Linux / Cloud)
```bash
gunicorn --workers 2 --bind 0.0.0.0:5000 backend.app:app
```

### Running with Waitress (Windows Production)
```bash
waitress-serve --port=5000 backend.app:app
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

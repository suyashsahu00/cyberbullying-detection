# Cyberbullying Detection & Explainability Tool 🛡️

A clean, minimal, NLP-powered web application for detecting cyberbullying in social media comments and tweets. Supports multilingual text (**English** & **Hinglish**), categorizes target types (**Age**, **Gender**, **Ethnicity**, **Religion**, **Other**), and provides word-level explainability by highlighting trigger words in red.

---

## 🌟 Key Features

- 🎯 **Instant Detection Verdict**: Displays clear badge — `Cyberbullying Detected` (red) vs `Not Cyberbullying` (green).
- 🏷️ **Target Category Classification**: Identifies targeted demographic bucket (`Age`, `Gender`, `Ethnicity`, `Religion`, `Other`).
- 📊 **Confidence Score**: Visual animated progress bar representing model prediction certainty.
- 💡 **Trigger Word Explainability**: Re-displays original input text with flagged trigger words highlighted/underlined in red.
- 🌐 **Automatic Language Indicator**: Automatically detects `English` vs `Hinglish` (code-switched Hindi-English).
- 🎨 **Calm, Trustworthy Aesthetic**: Built with HTML5, Bootstrap 5, FontAwesome, and custom slate/indigo styling.
- ⚡ **Preset Samples**: One-click preset buttons to instantly test various bullying categories.

---

## 📁 Repository Structure

```
cyberbullying-detection/
├── data/
│   ├── raw/                  # Original raw CSV datasets
│   └── processed/            # Cleaned, category-mapped datasets
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Text cleaning & language tagging
│   ├── 03_baseline_model.ipynb # TF-IDF + Logistic Regression baseline
│   ├── 04_muril_finetune.ipynb # MuRIL transformer fine-tuning (Colab GPU)
│   └── 05_evaluation_shap.ipynb # SHAP explainability analysis
├── src/
│   ├── preprocessing.py      # Cleaning, language tagging (English/Hinglish)
│   ├── model.py               # ML classifier pipeline & fallback rules
│   └── explainability.py      # Trigger word extraction & HTML highlighter
├── backend/
│   ├── app.py                 # Flask REST API & Web UI router
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── templates/
│   │   └── index.html         # Main Web UI template
│   └── static/
│       ├── css/               # Style tokens, glassmorphism, responsive design
│       └── js/                # Interactive UI event handlers & AJAX calls
├── models/                    # Saved model weights (.pkl, .pt)
├── report/                    # Project synopsis & documentation
├── .gitignore
├── pyproject.toml             # uv dependency specification
└── README.md
```

---

## 🚀 Quick Setup & Installation

### Option 1: Using `uv` (Recommended — 10x Fast Setup)

```bash
# 1. Initialize & install dependencies with uv
uv sync

# 2. Run the Flask Web Application
uv run python backend/app.py
```

### Option 2: Using standard `venv` & `pip`

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run the Flask Web Application
python backend/app.py
```

---

## 🖥️ Running the Application

After starting the server, open your browser and navigate to:

👉 **`http://127.0.0.1:5000`**

### API Endpoints

- `GET /` — Serves the main Web UI.
- `GET /api/health` — Health status check.
- `POST /api/analyze` — Analyzes text.
  **Request Body:**
  ```json
  {
    "text": "Go back to the kitchen and make me a sandwich bitch."
  }
  ```
  **Response:**
  ```json
  {
    "verdict": "Cyberbullying Detected",
    "is_cyberbullying": true,
    "category": "Gender",
    "confidence": 95.8,
    "language": "English",
    "explainability": {
      "trigger_words": ["bitch", "kitchen"],
      "highlighted_text": "Go back to the <mark class=\"trigger-highlight\">kitchen</mark> and make me a sandwich <mark class=\"trigger-highlight\">bitch</mark>."
    }
  }
  ```

---

## 🔬 Training & Notebooks

Run Jupyter Notebooks to explore dataset analysis, preprocessing, baseline model training, and transformer fine-tuning:

```bash
uv run jupyter notebook
# OR
jupyter notebook
```
Open notebooks sequentially from `notebooks/01_eda.ipynb` to `notebooks/05_evaluation_shap.ipynb`.

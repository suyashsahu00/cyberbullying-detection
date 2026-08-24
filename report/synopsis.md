# Project Report & Synopsis — Cyberbullying Detection Tool

## Abstract
Cyberbullying on social media platforms is a growing concern affecting digital well-being. This project delivers an AI-driven, transparent cyberbullying detection tool supporting English and Hinglish (code-switched Hindi-English) text.

## Core Features
1. **Multilingual Classification**: Categorizes content into target buckets — `Age`, `Gender`, `Ethnicity`, `Religion`, `Other`, or `Not Cyberbullying`.
2. **Explainability**: Highlights trigger words and phrases in red to provide user transparency on model verdicts.
3. **Automatic Language Detection**: Identifies whether comments are written in standard English or Hinglish.
4. **Clean Web UI**: Trustworthy, minimal interface built with Bootstrap 5 and custom CSS.

## System Architecture
- **Backend API**: Flask (`backend/app.py`)
- **NLP Engine**: Preprocessing, TF-IDF / Transformer classifier, Keyword-based Trigger Detection wrapper (`src/`)
- **Frontend**: HTML5, Bootstrap 5, FontAwesome, Vanilla JS (`frontend/`)

# FC 25 Steam Reviews Sentiment Analysis

This project implements a sentiment analysis pipeline for FC 25 Steam reviews. It combines domain-aware text preprocessing with a tuned Complement Naive Bayes model to better handle gaming-language patterns such as idioms, negation, and double negation.

## Overview

The system is designed to classify review sentiment reliably in noisy, informal text. The pipeline includes:

- Ground-truth label generation using multiple LLM prompting strategies and majority voting.
- Multi-layer preprocessing for sentiment-critical linguistic structures.
- A Complement Naive Bayes classifier tuned with `GridSearchCV`.
- A Streamlit app for real-time inference.

## Key Features

### 1. Triple-Prompt Ground Truth

Labels are generated with Google Gemini 1.5 Flash using three distinct prompt variants. Final labels are selected via majority voting to improve annotation consistency.

### 2. Multi-Layer Text Preprocessing

- **Layer 1: Idiom normalization**  
  Maps gaming idioms and colloquial phrases to sentiment-consistent forms (example: `can't wait` -> `very excited`).

- **Layer 2: Double-negation handling**  
  Detects and resolves patterns like negation + negation (example: `can't not recommend` -> `definitely recommend`).

- **Layer 3: Negation scope tagging**  
  Applies a 4-word sliding window to tag tokens under negation influence (example: `not good_NEG at_NEG all_NEG`).

### 3. Model Optimization

The baseline Naive Bayes approach is upgraded to `ComplementNB` and tuned using `GridSearchCV`, which improves performance on imbalanced text data.

### 4. Production-Ready Inference UI

A Streamlit dashboard provides local, low-latency sentiment predictions using the trained pipeline bundle.

## Repository Structure

| File               | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `streamlit_app.py` | Streamlit application for sentiment inference.                                 |
| `model.pkl`        | Serialized model bundle (pipeline, label encoder, and preprocessor artifacts). |
| `labeled_data.csv` | Labeled dataset used for model development and evaluation.                     |
| `requirements.txt` | Python dependencies for running the project.                                   |
| `readme.md`        | Project documentation.                                                         |

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis-fc25.git
cd sentiment-analysis-fc25/repo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python -m streamlit run streamlit_app.py
```

## Model Performance

The best-performing setup was the full multi-layer preprocessor on raw text.

| Version                       |   Accuracy | Improvement vs Baseline |
| ----------------------------- | ---------: | ----------------------: |
| Baseline (NB + BoW)           |     81.03% |                       - |
| GridSearch Optimized          |     86.21% |                  +5.18% |
| Simple Negation Handling      |     87.93% |                  +6.90% |
| Full Multi-Layer Preprocessor | **89.66%** |              **+8.63%** |

## Technical Notes

- **Text representation:** Bag-of-Words with `ngram_range=(1, 3)` and `char_wb` analyzer to capture phrase context and spelling variation.
- **Annotation quality control:** Inter-annotator consistency assessed with Fleiss' Kappa.
- **Deployment model:** Local Streamlit frontend calling the serialized ML pipeline for fast predictions.

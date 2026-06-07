# 📧 Spam Detection — ML & Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20SMS%20Spam-blue)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

A complete end-to-end spam detection system comparing classical Machine Learning and Deep Learning approaches on the **UCI SMS Spam Collection** dataset — from raw text to trained models with full evaluation and visual analysis.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline](#-pipeline)
- [Models](#-models)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Dataset](#-dataset)
- [Key Findings](#-key-findings)
- [Author](#-author)

---

## 🎯 Overview

This project builds a robust spam classifier using two parallel tracks:

- **Classical ML** — fast, interpretable models using TF-IDF features
- **Deep Learning** — sequence models that capture context and word order

Both tracks are evaluated side-by-side with consistent metrics (F1, ROC-AUC, Precision, Recall) and an **ensemble voting** system combines all models for the final prediction.

---

## 🔄 Pipeline

```
Raw Text
   │
   ▼
┌─────────────────────────────────┐
│         EDA & Analysis          │
│  class dist · word clouds ·     │
│  length stats · correlations    │
└────────────────┬────────────────┘
                 │
   ┌─────────────▼─────────────┐
   │      Text Preprocessing   │
   │  lowercase · URL tokens · │
   │  stopwords · lemmatize    │
   └──────┬──────────┬─────────┘
          │          │
   ┌──────▼──┐  ┌────▼────────┐
   │  TF-IDF │  │  Tokenizer  │
   │ (8K,    │  │ + Padding   │
   │  1-2gr) │  │ (seq len=120│
   └──────┬──┘  └────┬────────┘
          │          │
   ┌──────▼──┐  ┌────▼────────┐
   │Classical│  │    Deep     │
   │   ML    │  │  Learning   │
   │ NB·LR·  │  │ LSTM·BiLSTM │
   │ SVM·RF· │  │  CNN-LSTM   │
   │ XGBoost │  └────┬────────┘
   └──────┬──┘       │
          └─────┬────┘
                │
        ┌───────▼──────┐
        │   Ensemble   │
        │  (majority   │
        │    vote)     │
        └───────┬──────┘
                │
        ┌───────▼──────┐
        │  Evaluation  │
        │ F1·AUC·CM·   │
        │ ROC · errors │
        └──────────────┘
```

---

## 🤖 Models

### Classical ML (TF-IDF features)

| Model | Notes |
|-------|-------|
| Multinomial Naive Bayes | Strong baseline for text classification |
| Logistic Regression | Fast, interpretable, excellent precision |
| Linear SVM | Best classical model for high-dim text |
| Random Forest | Ensemble tree, captures non-linear patterns |
| XGBoost | Gradient boosting with class weight balancing |

### Deep Learning (Embedding + Sequence)

| Model | Architecture |
|-------|-------------|
| LSTM | 2-layer stacked LSTM with dropout |
| BiLSTM | Bidirectional LSTM — captures forward & backward context |
| CNN-LSTM | Conv1D for local features → BiLSTM for sequence context |

---

## 📊 Results

> Results may vary slightly depending on random seed and environment.

| Rank | Model | F1 | ROC-AUC | Precision | Recall |
|------|-------|----|---------|-----------|--------|
| 🥇 | BiLSTM | ~0.982 | ~0.998 | ~0.975 | ~0.989 |
| 🥈 | CNN-LSTM | ~0.979 | ~0.997 | ~0.971 | ~0.987 |
| 🥉 | Linear SVM | ~0.976 | ~0.997 | ~0.983 | ~0.969 |
| 4 | Logistic Regression | ~0.973 | ~0.995 | ~0.979 | ~0.967 |
| 5 | LSTM | ~0.971 | ~0.996 | ~0.963 | ~0.979 |
| 6 | XGBoost | ~0.968 | ~0.993 | ~0.971 | ~0.965 |
| 7 | Random Forest | ~0.961 | ~0.990 | ~0.977 | ~0.945 |
| 8 | Naive Bayes | ~0.944 | ~0.985 | ~0.939 | ~0.949 |

**Key insight:** Deep learning models slightly outperform classical ML, but Linear SVM and Logistic Regression are surprisingly competitive — and run in milliseconds.

---

## 📁 Project Structure

```
spam-detection-ml-dl/
│
├── 📓 spam_detection.ipynb     ← main notebook (all code)
├── 📄 README.md
├── 📋 requirements.txt
├── 🛡️  .gitignore
│
├── 📁 data/
│   └── SMSSpamCollection       ← dataset (auto-downloaded if missing)
│
└── 📁 outputs/                 ← generated plots
    ├── eda_report.png
    ├── tfidf_features.png
    ├── cm_ml_models.png
    ├── dl_training_curves.png
    ├── model_comparison.png
    └── roc_curves.png
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-detection-ml-dl.git
cd spam-detection-ml-dl
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook spam_detection.ipynb
```

### 5. Run all cells

The notebook will **automatically download** the dataset from UCI on first run. No manual setup needed.

---

## 📂 Dataset

**SMS Spam Collection** — UCI Machine Learning Repository

| Property | Value |
|----------|-------|
| Total messages | 5,574 |
| Ham (legitimate) | 4,827 (86.6%) |
| Spam | 747 (13.4%) |
| Language | English |
| Source | UCI ML Repository |

> Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering.* ACM DOCENG.

---

## 🔍 Key Findings

- **Spam messages are ~2× longer** on average and contain significantly more digits and special characters (`!`, `£`, `$`)
- **TF-IDF bigrams** (e.g. `call now`, `win cash`, `free entry`) are the most discriminative features
- **Bidirectional context** in BiLSTM helps catch subtle spam patterns missed by unidirectional models
- **Class imbalance** (87% ham) is handled via `class_weight='balanced'` (ML) and `class_weight={1: 5}` (DL)
- **Ensemble voting** across all 8 models reduces individual model errors and improves robustness
- For **production deployment**: Linear SVM or Logistic Regression offer the best speed/accuracy tradeoff
- For **maximum accuracy**: BiLSTM or CNN-LSTM ensemble is recommended

---

## 🛠️ Tech Stack

- **Data:** `pandas`, `numpy`
- **NLP:** `nltk` (tokenization, stopwords, lemmatization)
- **Features:** `scikit-learn` TF-IDF
- **ML:** `scikit-learn`, `xgboost`
- **DL:** `tensorflow` / `keras`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`

---

## 👤 Author

**Kasra** — Physics Student, Tabari University

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

Dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) and is used for academic/educational purposes.

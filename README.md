# Natural Language Processing with Disaster Tweets

This project uses **Natural Language Processing (NLP)** techniques to classify whether tweets are about real disasters or not. The goal is to build a machine learning model that can accurately distinguish between tweets announcing a real disaster (e.g., _"Forest fire near La Ronge Sask. Canada"_) and those that are not, but may use disaster-related words in a metaphorical sense (e.g., _"This concert is 🔥"_).

This project is based on the [Kaggle competition: Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Text Preprocessing](#2-text-preprocessing)
  - [3. Feature Extraction (Vectorization)](#3-feature-extraction-vectorization)
  - [4. Model Training and Evaluation](#4-model-training-and-evaluation)
  - [5. Submission](#5-submission)
- [Results](#results)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)

---

## 📌 Project Overview

The core task is a **binary text classification** problem. The model is trained on a dataset of over 7,000 tweets manually labeled as:

- `1` — Disaster-related tweet
- `0` — Not a disaster

This notebook walks through the entire NLP pipeline:

- Data cleaning and preprocessing
- Feature extraction
- Model training
- Submission file generation

A key feature of this project is the **extensive text preprocessing pipeline**, including spelling correction using `symspellpy`.

---

## 🔍 Methodology

### 1. Data Preparation

- Load `train.csv` and `test.csv` using `pandas`
- Drop irrelevant columns: `id`, `keyword`, and `location`
- Focus on `text` and `target` columns

### 2. Text Preprocessing

A custom `preprocess_text()` function is applied:

- ✅ Lowercase conversion
- 🚫 Remove:
  - HTML tags
  - URLs
  - Punctuation and special characters
  - Numbers
  - Emojis
- 🧹 Stopword removal using NLTK
- ✅ Spelling correction using `symspellpy`
- ✂️ Tokenization using `nltk.word_tokenize`
- 🔄 Lemmatization using NLTK's `WordNetLemmatizer`

### 3. Feature Extraction (Vectorization)

Two vectorization methods were explored:

- **CountVectorizer (Bag-of-Words)**
  - n-gram range: (1, 3)
- **TfidfVectorizer (TF-IDF)**
  - n-gram range: (1, 3)

### 4. Model Training and Evaluation

Models were evaluated using the **F1-Score**:

| Model                   | Vectorizer      | Parameters           | F1-Score |
|------------------------|-----------------|----------------------|----------|
| Logistic Regression     | CountVectorizer | ngram_range=(1, 3)   | ~0.73    |
| Multinomial Naive Bayes | CountVectorizer | ngram_range=(1, 3)   | ~0.76    |
| Logistic Regression     | TfidfVectorizer | ngram_range=(1, 3)   | ~0.74    |
| Multinomial Naive Bayes | TfidfVectorizer | ngram_range=(1, 3)   | ~0.73    |

Other models like SVM, Decision Trees, and Random Forests are included but commented out.

✅ **Best Model:** Multinomial Naive Bayes with `CountVectorizer`

### 5. Submission

- Fit `CountVectorizer` on full training data
- Train `MultinomialNB` model
- Preprocess and transform test data
- Predict and save results to `submission_1.csv` in the format: `id, target`

---

## ✅ Results

The final model:
- **Multinomial Naive Bayes** trained on **Bag-of-Words** features
- Showed strong performance on validation data
- **Spelling correction and thorough preprocessing** were crucial for improved accuracy

---

## ⚙️ How to Use

### On Kaggle

This notebook is designed to run in the Kaggle environment with the dataset readily available.

### Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

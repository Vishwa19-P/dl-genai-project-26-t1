# Smart MCQ Solver Challenge

A machine learning project developed for the **Smart MCQ Solver Challenge**, focusing on predicting the correct answer for multiple-choice questions using text-based machine learning techniques, pretrained language models, and ensemble learning.

---

## Project Overview

The objective of this project is to rank the five answer options (A–E) for each question and predict the most likely correct answer. Multiple machine learning approaches were explored, compared, and combined to improve prediction performance.

The project follows a complete machine learning workflow, including:

- Exploratory Data Analysis (EDA)
- Text preprocessing
- Feature engineering
- Feature selection
- Model development
- Cross-validation
- Ensemble learning
- Experiment tracking
- Kaggle submission and evaluation

---

## Models Implemented

The following models were developed and evaluated during the project:

- Scratch Logistic Ranking Model
- Complement Naive Bayes
- Pretrained Sentence Transformer (all-MiniLM-L6-v2)
- Rank SGD Classifier
- Question-level TF-IDF + TruncatedSVD + Logistic Regression

The final solution combines multiple models through weighted ensembling to generate the final predictions.

---

## Repository Structure

```
dl-genai-project-26-t1/
│
├── notebooks/
│   └── Smart_MCQ_Solver_Final.ipynb
│
├── scripts/
│   └── (Reserved for standalone Python scripts)
│
├── data/
│   └── (Dataset provided through Kaggle)
│
└── README.md
```

---

## Technologies Used

- Python
- Scikit-learn
- Sentence Transformers
- NumPy
- Pandas
- Weights & Biases (WandB)
- Kaggle Notebooks
- Git & GitHub

---

## Experiment Tracking

Experiments were tracked using **Weights & Biases (WandB)** to compare different models, validation metrics, and ensemble configurations.

Model evaluation and final submissions were performed through Kaggle.

---

## Results

Several ensemble configurations were evaluated during development. The final notebook includes:

- Individual model evaluation
- Cross-validation analysis
- Ensemble comparison
- Kaggle submission analysis

The best-performing ensemble was selected after comparing both offline validation metrics and public leaderboard performance.

---

## Author

**Vishwa Patel**

B.S. in Data Science and Applications, IIT Madras  
B.Tech. in Computer Engineering, Dharmsinh Desai University (DDU)

GitHub: https://github.com/Vishwa19-P

# 🧠 Smart MCQ Solver

A competition-trained multiple-choice question ranking system with a **knowledge-assisted Streamlit demo** for general MCQs.

The project was developed for the IIT Madras BS Degree **DL/GenAI project** and combines classical NLP ranking models, engineered lexical features, a pretrained MiniLM semantic baseline, and a weighted ensemble. The final competition pipeline was evaluated with grouped validation and submitted to Kaggle. A separate knowledge-assisted deployment layer uses public Wikipedia evidence and MiniLM semantic similarity to make the interactive demo more useful on arbitrary questions.

---

## 🚀 Live Demo

**Streamlit:** https://smart-mcq-solver-vishwa.streamlit.app

Enter one question and five answer options. The application returns the **top 3 ranked options**, score breakdowns, and the public knowledge sources used by the demo.

> **Important:** The deployed Streamlit application is a portfolio/demo layer. The official Kaggle submission remains the competition-trained v4 ensemble documented in the notebook.

---

## 📌 Project Overview

### Problem

Given a multiple-choice question and five candidate answers, rank the options and return the top three predictions.

### Objective

Build an end-to-end MCQ ranking pipeline that:

- preprocesses question and option text,
- constructs candidate-level lexical and statistical features,
- trains multiple ranking/classification models,
- evaluates models using question-grouped validation,
- tracks experiments with Weights & Biases (W&B),
- combines complementary signals into a final ensemble,
- generates a Kaggle submission, and
- exposes the project through an interactive Streamlit application.

---

## 🏗️ End-to-End Architecture

```text
                    ┌──────────────────────┐
                    │   MCQ Question +     │
                    │   Five Answer       │
                    │   Options            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Normalization   │
                    │ + Feature Engineering│
                    └──────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
       Logistic Ranking   Complement NB     Rank SGD
              │                │                 │
              └────────────────┼─────────────────┘
                               │
                               ▼
                    Question-level TF-IDF
                         Classifier
                               │
                               ▼
                       Heuristic Signal
                               │
                               ▼
                    Per-question normalization
                               │
                               ▼
                    ┌──────────────────────┐
                    │   V4 Weighted        │
                    │   Ensemble           │
                    └──────────┬───────────┘
                               │
                               ▼
                       Top-3 Predictions

        Streamlit demo adds an independent knowledge layer:

        Question → Wikipedia Retrieval → MiniLM Semantic Score
                              │
                              ▼
                     Knowledge-assisted
                         Demo Ranking
```

---

## 📊 Dataset

The project uses the Kaggle **Smart MCQ Solver Challenge** dataset:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The task is a ranking problem: each question has five candidate options and the system must identify the strongest answers.

The dataset files are kept in the private repository for reproducibility.

---

## 🧹 Text Preprocessing

The pipeline applies consistent normalization to questions and options using a `normalize_text` function.

Main steps include:

1. Unicode normalization using NFKD.
2. Conversion to ASCII-compatible text.
3. Lowercasing.
4. Removal of non-alphanumeric characters.
5. Whitespace normalization.

This creates a consistent representation for lexical and statistical feature extraction.

---

## 🧩 Feature Engineering

The candidate-level feature set captures multiple forms of question-option compatibility.

### Lexical features

- token overlap
- Jaccard similarity
- prompt coverage
- option coverage
- IDF-weighted overlap
- rare-token overlap
- bigram overlap
- token uniqueness relative to the other options
- relative option length

### Linguistic / structural signals

- number consistency
- negation indicators
- prompt-option relationships
- question-level candidate structure

The IDF statistics are calculated from the prepared corpus so that rare, informative tokens receive greater weight than common tokens.

---

## 🤖 Models Explored

### 1. Logistic Ranking

A feature-based logistic ranking model predicts the probability that a candidate option is the correct answer.

### 2. Complement Naive Bayes

Complement Naive Bayes provides a different probabilistic view of candidate correctness and adds diversity to the ensemble.

### 3. Question-level TF-IDF Classifier

A question-level TF-IDF model predicts the correct option from the question text itself.

### 4. Rank SGD

A stochastic-gradient ranking model provides another independent ranking signal.

### 5. Question SVD + Logistic Regression

A dimensionality-reduced question representation was also explored during experimentation. Its signal was ultimately excluded from the final v4 blend.

### 6. Pretrained MiniLM

`all-MiniLM-L6-v2` was evaluated as a standalone semantic similarity baseline. It was **not included in the official v4 Kaggle blend**.

---

## 🧠 Final V4 Ensemble

The final code-backed competition submission combines five normalized signals:

| Component | Weight |
|---|---:|
| Feature-based Logistic Ranking | 0.33 |
| Complement Naive Bayes Ranking | 0.22 |
| Question-level TF-IDF Classifier | 0.30 |
| Hand-designed Heuristic | 0.10 |
| Rank SGD | 0.05 |

The final score is:

```text
0.33 × rank_logreg_score_norm
+ 0.22 × rank_nb_score_norm
+ 0.30 × question_prob_norm
+ 0.10 × heuristic_score_norm
+ 0.05 × rank_sgd_score_norm
```

The scores are normalized within each question before blending so that the component models operate on comparable scales.

The five options are then ranked and the top three are returned.

---

## 🔬 Validation Strategy

A major concern in MCQ ranking is question-level leakage: the five options belonging to the same question must not be split across training and validation folds.

The project therefore uses:

```python
GroupKFold(n_splits=5)
```

with the **question ID as the grouping variable**.

This ensures that all options from a particular question remain together in either the training or validation partition.

### Final v4 held-out validation

| Metric | Result |
|---|---:|
| Validation questions | 2,000 |
| Top-1 Accuracy | 0.9975 |
| Macro F1 | 0.9972 |
| MAP@3 | 0.9980 |

These are grouped local validation results and **must not be interpreted as Kaggle leaderboard scores**.

---

## 🏆 Kaggle Results

The final competition submission history produced the following confirmed public scores:

| Submission | Public Score |
|---|---:|
| Earlier submission | 0.75228 |
| V2 | 0.75685 |
| V3 — initial | 0.74937 |
| V3 — revised | **0.75810** |
| V4 | **0.75810** |

### Final selected submission

**`submission_main_model_v4.csv`**

- Public Score: **0.75810**
- Private Score: **0.74747**
- Selected for the final leaderboard evaluation: **Yes**

The v4 result is tied with the revised v3 submission for the best confirmed public score in the project.

---

## 📈 Experiment Tracking — Weights & Biases

The official W&B project is:

```text
25f2004951-t22026
```

The final project execution contains four relevant runs:

| Run | Purpose |
|---|---|
| `scratch-logistic-ranking` | Feature-based ranking model |
| `complement-nb-ranking` | Complement Naive Bayes model |
| `pretrained-minilm-semantic` | Standalone pretrained semantic baseline |
| `final-v4-ensemble-validation` | Final v4 grouped validation |

W&B authentication is handled through a secret-managed `WANDB_API_KEY`. **No API key is stored in this repository.**

---

## 🌐 Streamlit Knowledge-Assisted Demo

The deployed application is designed as a more practical portfolio interface than the raw competition pipeline.

It combines:

1. the competition-trained v4 ensemble as one internal signal,
2. public Wikipedia evidence retrieval,
3. pretrained `all-MiniLM-L6-v2` semantic similarity, and
4. a combined demo score.

The application displays:

- top 3 predictions,
- demo score,
- knowledge score,
- v4 ensemble contribution, and
- retrieved Wikipedia sources.

### Why a separate demo layer?

The official Kaggle model is optimized for the competition's data distribution. A random general-knowledge question can be outside that distribution. The deployed demo therefore adds external evidence and semantic retrieval instead of presenting the competition model as a general-purpose knowledge model.

This separation keeps the competition result reproducible while making the public demo substantially more useful.

---

## 🧪 Example

For an input such as:

```text
Question: Who invented Newton's laws?

A. Einstein
B. Newton
C. Schrodinger
D. Pauli
E. Vishwa
```

the knowledge-assisted demo retrieves relevant public Wikipedia pages such as **Newton's laws of motion** and **Isaac Newton**, and ranks **Option B — Newton** first.

---

## 📁 Repository Structure

```text
dl-genai-project-26-t1/
│
├── data/
│   ├── .gitkeep
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── .gitkeep
│   └── Smart_MCQ_Solver.ipynb
│
├── scripts/
│   └── .gitkeep
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vishwa19-P/dl-genai-project-26-t1.git
cd dl-genai-project-26-t1
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 🔐 Secrets and Security

The project does **not** store API keys in source code.

For W&B-enabled notebook execution, provide the API key through a secure environment such as Kaggle Secrets using:

```text
WANDB_API_KEY
```

Do not commit:

- API keys
- passwords
- `.env` files
- Streamlit secrets
- authentication tokens

These are excluded through `.gitignore` where appropriate.

---

## 🛠️ Technology Stack

- **Python**
- **Pandas / NumPy** — data preparation and numerical processing
- **scikit-learn** — feature engineering, classical ML, validation, metrics
- **Sentence Transformers** — pretrained MiniLM semantic embeddings
- **Requests** — knowledge retrieval
- **Streamlit** — interactive deployment
- **Weights & Biases** — experiment tracking
- **Kaggle** — competition evaluation
- **Git / GitHub** — version control and project management

---

## 📐 Evaluation Metrics

Because the competition requires ranking the answer options, **MAP@3** is an important metric.

Additional metrics used during model comparison and validation include:

- Top-1 Accuracy
- Macro F1
- MAP@3

MAP@3 rewards the system when the correct option appears near the top of the three returned predictions, which matches the project's top-three ranking objective.

---

## 🔎 Key Design Decisions

### Why GroupKFold?

All five options for one question are structurally related. Splitting them independently can leak question-specific information between training and validation. GroupKFold prevents this by keeping each question entirely within one fold.

### Why an ensemble?

The component models capture different signals. Combining normalized outputs can make the final ranking less dependent on any single model's behavior.

### Why not include MiniLM in v4?

MiniLM was evaluated as a standalone semantic baseline. The official v4 competition submission was kept as the documented five-signal classical/engineered ensemble. MiniLM is instead used in the separate knowledge-assisted deployment layer.

### Why add Wikipedia retrieval to the demo?

The competition model is optimized for its training distribution rather than arbitrary world knowledge. Retrieval supplies explicit evidence for general questions, while MiniLM provides semantic similarity between the question/evidence and candidate options.

---

## 📌 Reproducibility

The repository contains the final notebook, application code, dependency specification, and competition data used by the project.

The notebook documents:

- preprocessing,
- feature engineering,
- model training,
- validation,
- W&B tracking,
- ensemble construction,
- submission generation, and
- Kaggle evaluation.

The final competition pipeline is the **v4 ensemble** described above.

---

## 🚧 Limitations

- The official Kaggle model is optimized for the competition data distribution and is not a general-purpose LLM.
- Knowledge-assisted predictions depend on the availability and relevance of retrieved public sources.
- Wikipedia retrieval can fail or return insufficient evidence for highly specialized or ambiguous questions.
- The demo's score is a ranking signal, not a calibrated probability of correctness.
- Kaggle leaderboard performance and general-purpose demo performance are separate evaluation settings.

---

## 🔮 Future Work

Potential extensions include:

- learning ensemble weights from strictly grouped validation data,
- calibrating component scores before blending,
- stronger pretrained language models for semantic matching,
- improved retrieval and reranking,
- domain-specific knowledge sources,
- richer error analysis, and
- extracting the verified inference pipeline into a cleaner production architecture.

---

## 📚 Project Artifacts

- **GitHub:** https://github.com/Vishwa19-P/dl-genai-project-26-t1
- **Live Streamlit Demo:** https://smart-mcq-solver-vishwa.streamlit.app
- **W&B Project:** `25f2004951-t22026`
- **Competition:** Kaggle Smart MCQ Solver Challenge

---

## 👤 Author

**Vishwa Patel**

B.Tech Computer Engineering — Dharmsinh Desai University (DDU)

IIT Madras BS in Data Science

---

## ⭐ Project Status

**Completed** — competition pipeline, evaluation, experiment tracking, GitHub repository, and deployed knowledge-assisted Streamlit demo are finalized.

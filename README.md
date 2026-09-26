# Smart MCQ Solver

A machine learning project for ranking answer options for multiple-choice questions. Developed as part of the IIT Madras BS Data Science DL/GenAI project.

## Live Demo

[Streamlit App](https://smart-mcq-solver-vishwa.streamlit.app)

The application accepts a question with five answer options and returns the top three predicted options. The deployed demo combines the competition-trained model with semantic similarity and public knowledge retrieval.

## Project Overview

The objective of the project is to rank five candidate answers (A-E) for each question and return the top three options.

The project includes:

- Text preprocessing
- Feature engineering
- Multiple machine learning models
- Ensemble learning
- Grouped cross-validation
- Weights & Biases experiment tracking
- Kaggle submission and evaluation
- Streamlit deployment

## Dataset

The project uses the dataset provided for the Smart MCQ Solver Challenge:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The task is to rank the five answer options for each question.

## Approach

The text data was normalized and converted into features based on the relationship between each question and its answer options.

The engineered features include:

- Token overlap
- Jaccard similarity
- Prompt coverage
- Option coverage
- IDF-weighted overlap
- Rare-token overlap
- Bigram overlap
- Number consistency
- Negation features
- Token uniqueness
- Relative option length

Several models were trained and compared before creating the final ensemble.

## Models

The project explored the following models:

- Logistic Regression ranking
- Complement Naive Bayes
- Question-level TF-IDF classifier
- Rank SGD
- SVD + Logistic Regression
- Pretrained MiniLM (`all-MiniLM-L6-v2`)

The pretrained MiniLM model was evaluated separately as a semantic baseline.

## Final V4 Ensemble

The final competition submission uses five signals:

| Model / Signal | Weight |
| --- | ---: |
| Logistic Ranking | 0.33 |
| Complement Naive Bayes | 0.22 |
| Question-level TF-IDF | 0.30 |
| Heuristic Score | 0.10 |
| Rank SGD | 0.05 |

The normalized scores from these components are combined to rank the five answer options.

## Validation

Validation was performed using 5-fold GroupKFold based on question ID.

Final v4 validation results:

| Metric | Result |
| --- | ---: |
| Validation Questions | 2,000 |
| Top-1 Accuracy | 99.75% |
| Macro F1 | 99.72% |
| MAP@3 | 99.80% |

## Kaggle Results

| Submission | Public Score |
| --- | ---: |
| Earlier submission | 0.75228 |
| V2 | 0.75685 |
| V3 (initial) | 0.74937 |
| V3 (revised) | 0.75810 |
| V4 | 0.75810 |

The final selected V4 submission:

- Public Score: **0.75810**
- Private Score: **0.74747**
- Selected for final leaderboard evaluation: **Yes**

## Weights & Biases

The project uses Weights & Biases for experiment tracking.

Project:

`25f2004951-t22026`

Main runs:

- `scratch-logistic-ranking`
- `complement-nb-ranking`
- `pretrained-minilm-semantic`
- `final-v4-ensemble-validation`

## Streamlit Application

The deployed application provides a knowledge-assisted version of the MCQ solver.

It combines:

- The competition-trained v4 ensemble
- Wikipedia-based knowledge retrieval
- MiniLM semantic similarity

The application displays the top three predictions, score breakdowns, and retrieved knowledge sources.

The Streamlit application is intended as an interactive demonstration of the project, while the official Kaggle submission remains the v4 competition pipeline.

## Repository Structure

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

## Running the Project

### Clone the repository

```bash
git clone https://github.com/Vishwa19-P/dl-genai-project-26-t1.git
cd dl-genai-project-26-t1
```

### Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in the browser.

## Security

No API keys or passwords are stored in the source code.

If W&B is enabled when running the notebook, provide the key through a secure environment such as Kaggle Secrets:

```text
WANDB_API_KEY
```

Do not commit API keys, `.env` files, authentication tokens, or Streamlit secrets.

## Evaluation Metrics

The main competition metric is MAP@3 because the task requires the system to return the top three answer options.

I also used:

- Top-1 Accuracy
- Macro F1
- MAP@3

MAP@3 is particularly useful here because it considers whether the correct answer appears within the first three predictions and rewards better ranking.

## Limitations

The official competition model is not a general-purpose language model. Its performance depends on the data distribution used for the competition.

The knowledge-assisted demo also has limitations:

- retrieved sources may not always contain enough information
- Wikipedia retrieval can fail or return less relevant pages
- semantic similarity is not the same as factual reasoning
- demo scores should not be interpreted as calibrated probabilities

The Kaggle evaluation and the Streamlit demo are therefore treated as two different evaluation settings.

## Project Links

GitHub: https://github.com/Vishwa19-P/dl-genai-project-26-t1

Streamlit: https://smart-mcq-solver-vishwa.streamlit.app

W&B Project: `25f2004951-t22026`

Kaggle: Smart MCQ Solver Challenge

## Author

**Vishwa Patel**

BS in Data Science, IIT Madras

B.Tech in Computer Engineering, Dharmsinh Desai University (DDU)

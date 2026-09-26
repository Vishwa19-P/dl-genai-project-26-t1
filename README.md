# Smart MCQ Solver

A machine learning project for ranking answer options for multiple-choice questions. The project was developed as part of the IIT Madras BS Data Science DL/GenAI project and was evaluated through Kaggle.

The main goal was to build a system that takes a question with five answer options and ranks the options, returning the top three. I experimented with several text-based models and feature-based approaches before combining the strongest signals into the final v4 ensemble.

The project also includes a Streamlit demo. The competition model is kept as the main experimental result, while the deployed demo adds semantic similarity and public knowledge retrieval so that it can handle general questions more naturally.

## Live Demo

Streamlit: https://smart-mcq-solver-vishwa.streamlit.app

The demo accepts one question and five answer options and returns the top three predictions. It also shows the scores used for ranking and the public sources retrieved for the question.

## Project Objective

The task is to rank five candidate answers for each multiple-choice question.

The project covers the complete workflow:

- data preparation
- text normalization
- feature engineering
- model training
- model comparison
- grouped cross-validation
- ensemble learning
- experiment tracking with Weights & Biases
- Kaggle submission and evaluation
- Streamlit deployment

## Dataset

The project uses the dataset provided for the Kaggle Smart MCQ Solver Challenge:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Each question has five candidate options. The competition requires the system to rank the options and return the top three.

The dataset is included in the private repository because the application and notebook use it for reproducibility.

## Text Preprocessing

Questions and options are normalized before feature extraction.

The main preprocessing steps are:

1. Unicode normalization using NFKD.
2. Conversion to ASCII-compatible text.
3. Lowercasing.
4. Removal of non-alphanumeric characters.
5. Whitespace normalization.

This gives the different models a consistent text representation.

## Feature Engineering

I created features that measure how closely an option is related to the question.

The feature set includes:

- token overlap
- Jaccard similarity
- prompt coverage
- option coverage
- IDF-weighted overlap
- rare-token overlap
- bigram overlap
- token uniqueness compared with other options
- relative option length
- number consistency
- negation-related features

These features are calculated for each question-option pair and are used by the ranking models.

## Models Tried

### Logistic Ranking

A logistic regression model is trained on question-option features and produces a probability for each candidate.

### Complement Naive Bayes

Complement Naive Bayes provides a different probabilistic model and adds another independent signal to the ensemble.

### Question-level TF-IDF Model

A TF-IDF representation of the question is used to predict which option is most likely to be correct.

### Rank SGD

A stochastic-gradient ranking model was also used as part of the final ensemble.

### Question SVD + Logistic Regression

I also experimented with reducing the question representation using SVD followed by logistic regression. This was useful during experimentation but was not included in the final v4 blend.

### Pretrained MiniLM

`all-MiniLM-L6-v2` was evaluated as a standalone pretrained semantic baseline. It was not included in the official v4 Kaggle submission.

## Final V4 Ensemble

The final competition submission uses five signals:

| Model / Signal | Weight |
| --- | ---: |
| Logistic Ranking | 0.33 |
| Complement Naive Bayes | 0.22 |
| Question-level TF-IDF model | 0.30 |
| Hand-designed heuristic | 0.10 |
| Rank SGD | 0.05 |

Before combining the signals, the scores are normalized within each question.

The final score is:

```text
0.33 * Logistic Ranking
+ 0.22 * Complement Naive Bayes
+ 0.30 * Question-level TF-IDF
+ 0.10 * Heuristic
+ 0.05 * Rank SGD
```

The five options are then sorted by the final score and the top three are returned.

## Validation

A question has five related options, so splitting individual options between training and validation can cause leakage.

To avoid this, I used:

```python
GroupKFold(n_splits=5)
```

with the question ID as the group.

This keeps all five options belonging to the same question in the same fold.

### Final v4 held-out validation

| Metric | Result |
| --- | ---: |
| Validation questions | 2,000 |
| Top-1 Accuracy | 0.9975 |
| Macro F1 | 0.9972 |
| MAP@3 | 0.9980 |

These are local grouped validation results. They are not Kaggle leaderboard scores.

## Kaggle Results

The main submission versions produced these public scores:

| Submission | Public Score |
| --- | ---: |
| Earlier submission | 0.75228 |
| V2 | 0.75685 |
| V3 (initial) | 0.74937 |
| V3 (revised) | 0.75810 |
| V4 | 0.75810 |

The selected final v4 submission was:

`submission_main_model_v4.csv`

Its recorded scores were:

- Public Score: **0.75810**
- Private Score: **0.74747**
- Selected for final leaderboard evaluation: **Yes**

The v4 result is tied with the revised v3 submission for the best confirmed public score in this project.

## Weights & Biases

The final W&B project is:

`25f2004951-t22026`

The main runs are:

- `scratch-logistic-ranking`
- `complement-nb-ranking`
- `pretrained-minilm-semantic`
- `final-v4-ensemble-validation`

The W&B API key is not stored in the repository. It is supplied through a secret called:

`WANDB_API_KEY`

## Streamlit Demo

The deployed application is slightly different from the competition submission.

The official v4 model was built for the Kaggle dataset and its distribution. If an unrelated general-knowledge question is entered, the competition model alone may not have enough information to answer it correctly.

For the public demo, I added:

- Wikipedia-based knowledge retrieval
- MiniLM semantic similarity
- the v4 ensemble as one of the ranking signals

The demo combines these signals and displays:

- top three predictions
- demo score
- knowledge score
- v4 ensemble score
- retrieved knowledge sources

This keeps the competition pipeline separate while making the deployed application more useful for normal MCQ examples.

For example, a question such as:

```text
Who invented Newton's laws?

A. Einstein
B. Newton
C. Schrodinger
D. Pauli
E. Vishwa
```

is handled using retrieved evidence about Newton and semantic matching, with Option B ranked first in the tested demo.

## Repository Structure

```text
dl-genai-project-26-t1/
|
├── data/
|   ├── .gitkeep
|   ├── train.csv
|   ├── test.csv
|   └── sample_submission.csv
|
├── notebooks/
|   ├── .gitkeep
|   └── Smart_MCQ_Solver.ipynb
|
├── scripts/
|   └── .gitkeep
|
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

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

## Why GroupKFold?

The five options for a question are related to the same question. If they are split independently, information from one question can appear in both training and validation data.

Using the question ID as the group keeps all options from a question together and gives a more reliable validation setup.

## Why Use an Ensemble?

The different models capture different information from the text.

The logistic ranking model, Naive Bayes model, question-level classifier, heuristic features, and Rank SGD model do not make their decisions in exactly the same way. Combining their normalized scores gives the final model several different signals instead of relying on a single classifier.

## Limitations

The official competition model is not a general-purpose language model. Its performance depends on the data distribution used for the competition.

The knowledge-assisted demo also has limitations:

- retrieved sources may not always contain enough information
- Wikipedia retrieval can fail or return less relevant pages
- semantic similarity is not the same as factual reasoning
- demo scores should not be interpreted as calibrated probabilities

The Kaggle evaluation and the Streamlit demo are therefore treated as two different evaluation settings.

## Future Work

Some improvements I would consider next are:

- learning the ensemble weights instead of setting them manually
- calibrating the individual model scores
- improving retrieval and reranking
- testing stronger pretrained language models
- adding better error analysis
- separating the inference pipeline into smaller reusable modules
- evaluating the demo on a larger set of unseen questions

## Project Links

GitHub: https://github.com/Vishwa19-P/dl-genai-project-26-t1

Streamlit: https://smart-mcq-solver-vishwa.streamlit.app

W&B Project: `25f2004951-t22026`

Kaggle: Smart MCQ Solver Challenge

## Author

**Vishwa Patel**

BS in Data Science, IIT Madras

B.Tech in Computer Engineering, Dharmsinh Desai University (DDU)

## Project Status

Completed.

The final competition pipeline, Kaggle submission, W&B tracking, GitHub repository, and deployed Streamlit demo are in place.

import math
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

RANDOM_STATE = 42
OPTION_COLS = ["A", "B", "C", "D", "E"]
TARGET_COL = "answer"
NEGATION_WORDS = {"no", "not", "never", "none", "cannot", "can't", "except", "incorrect", "false", "least"}


def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", errors="ignore")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return [token for token in normalize_text(text).split() if token]


def bigrams(tokens):
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def has_negation(tokens):
    return int(any(token in NEGATION_WORDS for token in tokens))


def extract_numbers(text):
    return re.findall(r"\d+", str(text))


def prepare_base_frame(df):
    prepared = df.copy()
    prepared["prompt_clean"] = prepared["prompt"].map(normalize_text)

    for col in OPTION_COLS:
        prepared[f"{col}_clean"] = prepared[col].map(normalize_text)

    prepared["all_choices_text"] = prepared.apply(
        lambda row: " ".join([f"{col} {row[f'{col}_clean']}" for col in OPTION_COLS]),
        axis=1,
    )

    prepared["question_bundle"] = prepared.apply(
        lambda row: (
            f"prompt {row['prompt_clean']} "
            + " ".join([f"choice {col} {row[f'{col}_clean']}" for col in OPTION_COLS])
        ),
        axis=1,
    )
    return prepared


def build_idf_lookup(prepared_frames):
    document_frequency = Counter()
    total_documents = 0

    for frame in prepared_frames:
        prompt_sets = frame["prompt_clean"].map(lambda text: set(text.split()))
        total_documents += len(prompt_sets)
        for token_set in prompt_sets:
            document_frequency.update(token_set)

        for col in OPTION_COLS:
            option_sets = frame[f"{col}_clean"].map(lambda text: set(text.split()))
            total_documents += len(option_sets)
            for token_set in option_sets:
                document_frequency.update(token_set)

    return {
        token: math.log((1 + total_documents) / (1 + freq)) + 1.0
        for token, freq in document_frequency.items()
    }


def build_candidate_frame(df, idf_lookup):
    records = []

    for row in df.itertuples(index=False):
        prompt_clean = row.prompt_clean
        prompt_tokens = tokenize(prompt_clean)
        prompt_token_set = set(prompt_tokens)
        prompt_bigrams = bigrams(prompt_tokens)
        prompt_numbers = set(extract_numbers(prompt_clean))
        prompt_negation = has_negation(prompt_tokens)

        option_texts = {col: getattr(row, f"{col}_clean") for col in OPTION_COLS}
        option_tokens_map = {col: tokenize(text) for col, text in option_texts.items()}
        option_sets_map = {col: set(tokens) for col, tokens in option_tokens_map.items()}
        option_bigrams_map = {col: bigrams(tokens) for col, tokens in option_tokens_map.items()}
        option_numbers_map = {col: set(extract_numbers(option_texts[col])) for col in OPTION_COLS}

        option_lengths = {col: len(option_tokens_map[col]) for col in OPTION_COLS}
        avg_option_length = np.mean(list(option_lengths.values()))
        max_option_length = max(option_lengths.values())
        min_option_length = min(option_lengths.values())

        for option_idx, option in enumerate(OPTION_COLS):
            option_clean = option_texts[option]
            option_tokens = option_tokens_map[option]
            option_token_set = option_sets_map[option]
            option_bigrams = option_bigrams_map[option]
            option_numbers = option_numbers_map[option]

            shared_tokens = prompt_token_set & option_token_set
            union_tokens = prompt_token_set | option_token_set
            shared_bigrams = prompt_bigrams & option_bigrams
            weighted_overlap = sum(idf_lookup.get(token, 1.0) for token in shared_tokens)
            rare_overlap_count = sum(
                1 for token in shared_tokens if idf_lookup.get(token, 1.0) > 4.0
            )

            other_options = [col for col in OPTION_COLS if col != option]
            other_sets = [option_sets_map[col] for col in other_options]
            other_unions = set().union(*other_sets) if other_sets else set()
            unique_vs_others = option_token_set - other_unions
            shared_with_others = [
                len(option_token_set & option_sets_map[col]) for col in other_options
            ]

            record = {
                "question_id": row.id,
                "option": option,
                "pair_text": f"{prompt_clean} [sep] option {option} [sep] {option_clean}",
                "question_bundle": row.question_bundle,
                "option_text": option_clean,
                "prompt_word_count": len(prompt_tokens),
                "option_word_count": len(option_tokens),
                "overlap_count": len(shared_tokens),
                "jaccard_overlap": safe_div(len(shared_tokens), len(union_tokens)),
                "prompt_coverage": safe_div(len(shared_tokens), len(prompt_token_set)),
                "option_coverage": safe_div(len(shared_tokens), len(option_token_set)),
                "weighted_overlap": weighted_overlap,
                "rare_overlap_count": rare_overlap_count,
                "bigram_overlap_count": len(shared_bigrams),
                "number_overlap_count": len(prompt_numbers & option_numbers),
                "number_mismatch_flag": int(bool(prompt_numbers) and not bool(prompt_numbers & option_numbers)),
                "prompt_has_negation": prompt_negation,
                "option_has_negation": has_negation(option_tokens),
                "negation_match": int(prompt_negation == has_negation(option_tokens)),
                "token_uniqueness_vs_others": len(unique_vs_others),
                "shared_with_others_mean": float(np.mean(shared_with_others)) if shared_with_others else 0.0,
                "shared_with_others_max": float(np.max(shared_with_others)) if shared_with_others else 0.0,
                "relative_option_length": safe_div(len(option_tokens), avg_option_length),
                "is_longest_option": int(option_lengths[option] == max_option_length),
                "is_shortest_option": int(option_lengths[option] == min_option_length),
                "option_index": option_idx,
            }
            if hasattr(row, TARGET_COL):
                record["target"] = int(getattr(row, TARGET_COL) == option)
            records.append(record)

    return pd.DataFrame(records)


numeric_features = [
    "prompt_word_count", "option_word_count", "overlap_count", "jaccard_overlap",
    "prompt_coverage", "option_coverage", "weighted_overlap", "rare_overlap_count",
    "bigram_overlap_count", "number_overlap_count", "number_mismatch_flag",
    "prompt_has_negation", "option_has_negation", "negation_match",
    "token_uniqueness_vs_others", "shared_with_others_mean", "shared_with_others_max",
    "relative_option_length", "is_longest_option", "is_shortest_option", "option_index",
]
rank_feature_columns = ["pair_text", "question_bundle", "option_text"] + numeric_features
question_feature_column = "question_bundle"


def build_rank_logreg_pipeline():
    feature_builder = ColumnTransformer(
        transformers=[
            ("pair_word_tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.97, sublinear_tf=True), "pair_text"),
            ("pair_char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True), "pair_text"),
            ("option_word_tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True), "option_text"),
            ("numeric", MaxAbsScaler(), numeric_features),
        ], remainder="drop", sparse_threshold=1.0,
    )
    return Pipeline([
        ("features", feature_builder),
        ("select", SelectPercentile(score_func=chi2, percentile=55)),
        ("classifier", LogisticRegression(C=2.5, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)),
    ])


def build_rank_nb_pipeline():
    feature_builder = ColumnTransformer(
        transformers=[
            ("pair_word_tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True), "pair_text"),
            ("question_word_tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True), "question_bundle"),
            ("numeric", MaxAbsScaler(), numeric_features),
        ], remainder="drop", sparse_threshold=1.0,
    )
    return Pipeline([
        ("features", feature_builder),
        ("select", SelectPercentile(score_func=chi2, percentile=40)),
        ("classifier", ComplementNB(alpha=0.35)),
    ])


def build_question_multiclass_pipeline():
    return Pipeline([
        ("features", ColumnTransformer(
            transformers=[
                ("bundle_word_tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 3), min_df=2, max_df=0.97, sublinear_tf=True), "question_bundle"),
                ("bundle_char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True), "question_bundle"),
            ], remainder="drop", sparse_threshold=1.0,
        )),
        ("select", SelectPercentile(score_func=chi2, percentile=65)),
        ("classifier", OneVsRestClassifier(
                LogisticRegression(
                    C=3.0,
                    class_weight="balanced",
                    max_iter=4000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                )
            )),
    ])


def build_rank_sgd_pipeline():
    feature_builder = ColumnTransformer(
        transformers=[
            ("pair_word_tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.97, sublinear_tf=True), "pair_text"),
            ("option_word_tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True), "option_text"),
            ("numeric", MaxAbsScaler(), numeric_features),
        ], remainder="drop", sparse_threshold=1.0,
    )
    return Pipeline([
        ("features", feature_builder),
        ("select", SelectPercentile(score_func=chi2, percentile=50)),
        ("classifier", SGDClassifier(loss="modified_huber", alpha=1e-4, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_question_probability_frame(question_ids, probability_matrix):
    rows = []
    for qid, probs in zip(question_ids, probability_matrix):
        for option_idx, option in enumerate(OPTION_COLS):
            rows.append({"question_id": qid, "option": option, "question_prob": probs[option_idx]})
    return pd.DataFrame(rows)


def minmax_normalize_by_question(df, score_columns):
    normalized = df.copy()
    for score_column in score_columns:
        group_min = normalized.groupby("question_id")[score_column].transform("min")
        group_max = normalized.groupby("question_id")[score_column].transform("max")
        normalized[f"{score_column}_norm"] = np.where(
            (group_max - group_min) > 0,
            (normalized[score_column] - group_min) / (group_max - group_min),
            0.0,
        )
    return normalized


def train_models(train, test):
    train_prepared = prepare_base_frame(train)
    test_prepared = prepare_base_frame(test)
    idf_lookup = build_idf_lookup([train_prepared, test_prepared])
    train_long = build_candidate_frame(train_prepared, idf_lookup)

    label_to_idx = {label: idx for idx, label in enumerate(OPTION_COLS)}
    question_train = train_prepared[["id", "question_bundle", "answer"]].copy()
    question_train["label_idx"] = question_train["answer"].map(label_to_idx)

    logreg = build_rank_logreg_pipeline()
    logreg.fit(train_long[rank_feature_columns], train_long["target"])

    nb = build_rank_nb_pipeline()
    nb.fit(train_long[rank_feature_columns], train_long["target"])

    question_model = build_question_multiclass_pipeline()
    question_model.fit(question_train[[question_feature_column]], question_train["label_idx"])

    sgd = build_rank_sgd_pipeline()
    sgd.fit(train_long[rank_feature_columns], train_long["target"])

    return {
        "idf_lookup": idf_lookup,
        "logreg": logreg,
        "nb": nb,
        "question_model": question_model,
        "sgd": sgd,
    }


def predict_one(prompt, options, models):
    row = {"id": 1, "prompt": prompt, **dict(zip(OPTION_COLS, options))}
    frame = pd.DataFrame([row])
    prepared = prepare_base_frame(frame)
    candidates = build_candidate_frame(prepared, models["idf_lookup"])

    logreg_positive_index = list(models["logreg"].classes_).index(1)
    candidates["rank_logreg_score"] = models["logreg"].predict_proba(candidates[rank_feature_columns])[:, logreg_positive_index]

    nb_positive_index = list(models["nb"].classes_).index(1)
    candidates["rank_nb_score"] = models["nb"].predict_proba(candidates[rank_feature_columns])[:, nb_positive_index]

    question_probs = models["question_model"].predict_proba(prepared[[question_feature_column]])
    question_prob_frame = build_question_probability_frame([1], question_probs)
    candidates = candidates.merge(question_prob_frame, on=["question_id", "option"], how="left")

    sgd_positive_index = list(models["sgd"].classes_).index(1)
    candidates["rank_sgd_score"] = models["sgd"].predict_proba(candidates[rank_feature_columns])[:, sgd_positive_index]

    candidates["heuristic_score"] = (
        0.40 * candidates["weighted_overlap"]
        + 0.20 * candidates["prompt_coverage"]
        + 0.10 * candidates["option_coverage"]
        + 0.15 * candidates["bigram_overlap_count"]
        + 0.15 * candidates["token_uniqueness_vs_others"]
    )

    score_columns = ["rank_logreg_score", "rank_nb_score", "question_prob", "heuristic_score", "rank_sgd_score"]
    candidates = minmax_normalize_by_question(candidates, score_columns)

    candidates["blend_score_v4"] = (
        0.33 * candidates["rank_logreg_score_norm"]
        + 0.22 * candidates["rank_nb_score_norm"]
        + 0.30 * candidates["question_prob_norm"]
        + 0.10 * candidates["heuristic_score_norm"]
        + 0.05 * candidates["rank_sgd_score_norm"]
    )

    ranked = candidates.sort_values("blend_score_v4", ascending=False).reset_index(drop=True)
    return ranked


@st.cache_data

def load_train_data():
    return pd.read_csv("data/train.csv")


@st.cache_resource

def load_models():
    train = load_train_data()
    test = pd.read_csv("data/test.csv")
    return train_models(train, test)


st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="wide")

st.title("🧠 Smart MCQ Solver")
st.write("Enter a multiple-choice question and its five options. The final v4 ensemble ranks the options and returns the top 3 predictions.")
st.caption("Final v4 ensemble: Logistic Ranking + Complement NB + Question-level TF-IDF + Heuristic + Rank SGD")

with st.form("mcq_form"):
    prompt = st.text_area("Question", height=150, placeholder="Enter the MCQ question here...")
    cols = st.columns(2)
    options = []
    labels = OPTION_COLS
    for i, label in enumerate(labels):
        with cols[i % 2]:
            options.append(st.text_area(f"Option {label}", height=110, key=f"option_{label}"))
    if st.form_submit_button("🚀 Solve MCQ", type="primary"):
        if not prompt.strip() or any(not option.strip() for option in options):
            st.error("Please enter the question and all five options.")
        else:
            with st.spinner("Loading the v4 ensemble and ranking the options... The first run may take a few minutes."):
                try:
                    models = load_models()
                    ranked = predict_one(prompt, options, models)
                except Exception as error:
                    st.error(f"The model could not be loaded or executed: {type(error).__name__}: {error}")
                    st.stop()

            st.subheader("Top 3 Predictions")
            for rank, (_, row) in enumerate(ranked.head(3).iterrows(), start=1):
                st.success(f"#{rank} — Option {row['option']}  |  Ensemble score: {row['blend_score_v4']:.4f}")
                st.write(options[OPTION_COLS.index(row['option'])])

            with st.expander("Show all five ranked options"):
                display_df = ranked[["option", "blend_score_v4"]].copy()
                display_df.columns = ["Option", "V4 Ensemble Score"]
                display_df["V4 Ensemble Score"] = display_df["V4 Ensemble Score"].round(4)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Model selection and blend weights follow the final code-backed v4 pipeline in the project notebook.")

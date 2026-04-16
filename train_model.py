"""
Train TF-IDF + Logistic Regression on UpdatedResumeDataSet.csv and save a single artifact
used for both category classification and JD-based ranking (cosine similarity).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CSV = Path(__file__).resolve().parent / "UpdatedResumeDataSet.csv"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "Category" not in df.columns or "Resume" not in df.columns:
        raise ValueError("CSV must have columns: Category, Resume")
    df = df.dropna(subset=["Category", "Resume"])
    df["Resume"] = df["Resume"].astype(str).str.strip()
    df = df[df["Resume"].str.len() > 0]
    return df.reset_index(drop=True)


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    y = df["Category"]
    texts = df["Resume"]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Refit on full data for deployment
    X_full = vectorizer.fit_transform(texts)
    clf_full = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf_full.fit(X_full, y)

    bundle = {
        "vectorizer": vectorizer,
        "classifier": clf_full,
        "X_corpus": X_full,
        "categories": df["Category"].tolist(),
        "resume_texts": df["Resume"].tolist(),
        "holdout_accuracy": acc,
        "classification_report": report,
        "class_labels": clf_full.classes_.tolist(),
    }
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train resume NLP models")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to UpdatedResumeDataSet.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_DIR / "model_bundle.joblib",
        help="Output path for joblib bundle",
    )
    args = parser.parse_args()

    df = load_dataset(args.csv)
    print(f"Loaded {len(df)} resumes, {df['Category'].nunique()} categories")

    bundle = train(df)
    print(f"Hold-out accuracy (stratified): {bundle['holdout_accuracy']:.4f}")
    print(bundle["classification_report"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"Saved bundle to {args.out}")


if __name__ == "__main__":
    main()

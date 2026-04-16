"""
Streamlit app: upload resume text files and rank them against a job description.
Run: streamlit run app.py
Requires: python train_model.py (creates artifacts/model_bundle.joblib)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "artifacts" / "model_bundle.joblib"
DEFAULT_CSV = ROOT / "UpdatedResumeDataSet.csv"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _artifact_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else -1.0


@st.cache_resource
def load_bundle(path: Path, _mtime: float):
    return joblib.load(path)


@st.cache_resource
def load_semantic_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def classify_text(bundle, text: str) -> tuple[np.ndarray, np.ndarray]:
    vec = bundle["vectorizer"].transform([text])
    proba = bundle["classifier"].predict_proba(vec)[0]
    classes = bundle["classifier"].classes_
    order = np.argsort(proba)[::-1]
    return classes[order], proba[order]


def rank_uploaded_resumes(bundle, query: str, resumes: list[dict]) -> pd.DataFrame:
    semantic_model = load_semantic_model(SEMANTIC_MODEL_NAME)
    texts = [item["text"] for item in resumes]
    names = [item["name"] for item in resumes]
    embeddings = semantic_model.encode([query, *texts], normalize_embeddings=True)
    query_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]
    sims = resume_embeddings @ query_embedding

    rows = []
    for rank_idx in np.argsort(sims)[::-1]:
        resume_text = texts[rank_idx]
        classes, probas = classify_text(bundle, resume_text)
        rows.append(
            {
                "Resume file": names[rank_idx],
                "Match score": float(sims[rank_idx]),
                "Predicted category": classes[0],
                "Category confidence": float(probas[0]),
                "Resume preview": resume_text[:400] + ("…" if len(resume_text) > 400 else ""),
            }
        )

    df = pd.DataFrame(rows)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    df["Match score"] = df["Match score"].round(4)
    df["Category confidence"] = df["Category confidence"].round(4)
    return df


def classify_uploaded_resumes(
    bundle, resumes: list[dict], top_k: int = 3
) -> pd.DataFrame:
    rows = []
    for item in resumes:
        classes, probas = classify_text(bundle, item["text"])
        top_k = min(top_k, len(classes))
        top_roles = [
            f"{classes[i]} ({probas[i]:.1%})" for i in range(top_k)
        ]
        rows.append(
            {
                "Resume file": item["name"],
                "Predicted role": classes[0],
                "Confidence": float(probas[0]),
                f"Top {top_k} roles": "; ".join(top_roles),
                "Resume preview": item["text"][:400]
                + ("…" if len(item["text"]) > 400 else ""),
            }
        )

    df = pd.DataFrame(rows)
    df["Confidence"] = df["Confidence"].round(4)
    return df


def read_uploaded_text_files(files) -> list[dict]:
    resumes = []
    for uploaded in files:
        raw = uploaded.getvalue()
        text = raw.decode("utf-8", errors="ignore").strip()
        if text:
            resumes.append({"name": uploaded.name, "text": text})
    return resumes


def ensure_model(csv_path: Path) -> Path:
    """Train model if artifact missing."""
    import subprocess
    import sys

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "train_model.py"), "--csv", str(csv_path)],
        check=True,
        cwd=str(ROOT),
    )
    return BUNDLE_PATH


def main() -> None:
    st.set_page_config(
        page_title="Resume Screener",
        page_icon="📄",
        layout="wide",
    )
    st.title("Resume ranking for a job")
    st.caption(
        "Upload resume text files to predict roles directly or use semantic ranking to compare them against a job description by meaning, not just exact keywords."
    )

    csv_path = st.sidebar.text_input("Dataset CSV path", value=str(DEFAULT_CSV))
    csv_file = Path(csv_path).expanduser()
    if not csv_file.is_file():
        st.error(f"CSV not found: {csv_file}")
        st.stop()

    if BUNDLE_PATH.is_file():
        bundle = load_bundle(BUNDLE_PATH, _artifact_mtime(BUNDLE_PATH))
    else:
        bundle = None

    if bundle is None:
        st.warning("No trained model found. Training once (may take a minute)…")
        with st.spinner("Training…"):
            try:
                ensure_model(csv_file)
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()
        bundle = load_bundle(BUNDLE_PATH, _artifact_mtime(BUNDLE_PATH))
    else:
        if st.sidebar.button("Retrain on current CSV"):
            with st.spinner("Retraining…"):
                try:
                    ensure_model(csv_file)
                    st.cache_resource.clear()
                    bundle = load_bundle(BUNDLE_PATH, _artifact_mtime(BUNDLE_PATH))
                except Exception as e:
                    st.sidebar.error(str(e))

    st.sidebar.metric("Resumes in model", len(bundle["resume_texts"]))
    st.sidebar.metric("Categories", len(bundle["class_labels"]))
    if "holdout_accuracy" in bundle:
        st.sidebar.metric("Hold-out accuracy", f"{bundle['holdout_accuracy']:.1%}")

    tab_roles, tab_rank = st.tabs(
        ["Predict roles", "Semantic Ranking Mode (SMART MODE)"]
    )

    with tab_roles:
        st.subheader("Resume text files")
        top_k = st.slider("Show top K role predictions", 2, 5, 3)
        role_uploads = st.file_uploader(
            "Upload one or more `.txt` resume files to predict roles",
            type=["txt"],
            accept_multiple_files=True,
            key="role_uploads",
        )

        if st.button("Predict roles", key="predict_roles_btn"):
            if not role_uploads:
                st.warning("Upload at least one resume text file.")
                st.stop()

            resumes = read_uploaded_text_files(role_uploads)
            if not resumes:
                st.error("None of the uploaded files contained readable text.")
                st.stop()

            df_roles = classify_uploaded_resumes(bundle, resumes, top_k=top_k)
            st.dataframe(df_roles, use_container_width=True, hide_index=True)

            csv_bytes = df_roles.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download role predictions CSV",
                data=csv_bytes,
                file_name="resume_roles.csv",
                mime="text/csv",
            )

    with tab_rank:
        st.subheader("SMART MODE")
        st.caption(
            "Uses Sentence Transformers for semantic ranking, so similar meanings like `ML Engineer` and `Machine Learning Developer` can still match strongly."
        )
        st.subheader("Job description")
        jd = st.text_area(
            "Paste the job description or required skills",
            height=220,
            key="jd",
            placeholder="e.g. We need a Python developer with Django, REST APIs, PostgreSQL, and strong NLP experience.",
        )

        st.subheader("Resume text files")
        rank_uploads = st.file_uploader(
            "Upload one or more `.txt` resume files",
            type=["txt"],
            accept_multiple_files=True,
            key="rank_uploads",
        )

        if st.button("Rank candidates", key="rank_btn"):
            if not jd.strip():
                st.warning("Enter a job description first.")
                st.stop()
            if not rank_uploads:
                st.warning("Upload at least one resume text file.")
                st.stop()

            resumes = read_uploaded_text_files(rank_uploads)
            if not resumes:
                st.error("None of the uploaded files contained readable text.")
                st.stop()

            df_rank = rank_uploaded_resumes(bundle, jd, resumes)
            best = df_rank.iloc[0]

            st.success(
                f"Best candidate: **{best['Resume file']}** with semantic match score **{best['Match score']:.4f}**"
            )
            st.write(
                f"Most likely role: **{best['Predicted category']}** "
                f"({best['Category confidence']:.1%} confidence)"
            )

            st.dataframe(df_rank, use_container_width=True, hide_index=True)

            csv_bytes = df_rank.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download ranking CSV",
                data=csv_bytes,
                file_name="resume_ranking.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

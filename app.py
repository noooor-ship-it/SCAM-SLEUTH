import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="SCAM SLEUTH | Job Scam Screening",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #10243a, #174b66);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin-bottom: .25rem;}
    .hero p {color: #d9e8f2; margin-bottom: 0;}
    .small-note {color: #65758b; font-size: .9rem;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5eaf0;
        padding: 12px; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛡️ SCAM SLEUTH</h1>
  <p>Intelligent Fake Job & Internship Detection Using Data Mining</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Academic prototype: predictions identify patterns learned from the uploaded "
    "historical dataset. They do not prove that an employer is fraudulent."
)

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

def find_text(row, cols):
    parts = []
    for col in cols:
        value = row.get(col, "")
        if pd.notna(value):
            parts.append(str(value))
    return " ".join(parts)

def make_text_frame(data):
    preferred = [
        "title", "company_profile", "description",
        "requirements", "benefits"
    ]
    cols = [c for c in preferred if c in data.columns]
    if not cols:
        cols = data.select_dtypes(include=["object"]).columns.tolist()
    if not cols:
        raise ValueError("No text columns were found in the dataset.")
    return data[cols].fillna("").astype(str).agg(" ".join, axis=1), cols

with st.sidebar:
    st.header("Project navigation")
    page = st.radio(
        "Choose a section",
        ["Overview", "Train & Evaluate", "Check a Job", "Data Mining Insights"]
    )
    st.divider()
    st.caption("SCAM SLEUTH • BCA Data Mining Mini Project")

st.subheader("1. Upload the job-postings dataset")
uploaded = st.file_uploader(
    "Upload the fake_job_postings.csv file",
    type=["csv"],
    help="Use the labelled fake job postings CSV with a 'fraudulent' target column."
)

if uploaded is not None:
    try:
        raw = load_csv(uploaded.getvalue())
        st.success(f"Dataset loaded: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
        with st.expander("Preview dataset"):
            st.dataframe(raw.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()
else:
    raw = None
    st.warning("Upload the CSV to activate model training, prediction and analysis.")

if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Mining", "Text + tabular")
    c2.metric("Classification", "Logistic Regression")
    c3.metric("Exploration", "K-Means + Apriori")
    st.markdown("""
    **Workflow**
    1. Load a labelled job-postings dataset.
    2. Inspect records, missing values and class distribution.
    3. Convert job-advertisement text into TF-IDF features.
    4. Train a Logistic Regression classifier using a held-out test set.
    5. Review precision, recall, F1-score and the confusion matrix.
    6. Enter a job advertisement for a model-generated screening result.
    7. Explore clusters and frequent feature combinations.
    """)
    st.caption("The app trains from the dataset you upload. It does not use invented model scores.")

if raw is not None:
    if "fraudulent" not in raw.columns:
        st.error("The uploaded dataset must contain a 'fraudulent' target column (0 = genuine, 1 = fraudulent).")
        st.stop()

    if page in ["Train & Evaluate", "Check a Job", "Data Mining Insights"]:
        if st.button("Train / refresh model", type="primary"):
            with st.spinner("Cleaning text and training the model..."):
                work = raw.copy()
                work = work.drop_duplicates()
                work["fraudulent"] = pd.to_numeric(work["fraudulent"], errors="coerce")
                work = work[work["fraudulent"].isin([0, 1])].copy()
                work["fraudulent"] = work["fraudulent"].astype(int)
                texts, text_cols = make_text_frame(work)
                y = work["fraudulent"]

                if y.nunique() < 2 or y.value_counts().min() < 2:
                    st.error("The dataset needs at least two examples in each target class.")
                    st.stop()

                X_train_text, X_test_text, y_train, y_test = train_test_split(
                    texts, y, test_size=0.2, random_state=42, stratify=y
                )
                vectorizer = TfidfVectorizer(
                    max_features=3000, stop_words="english",
                    ngram_range=(1, 2), min_df=2
                )
                X_train = vectorizer.fit_transform(X_train_text)
                X_test = vectorizer.transform(X_test_text)

                model = LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                st.session_state["model"] = model
                st.session_state["vectorizer"] = vectorizer
                st.session_state["text_cols"] = text_cols
                st.session_state["metrics"] = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision (fraud)": precision_score(y_test, pred, zero_division=0),
                    "Recall (fraud)": recall_score(y_test, pred, zero_division=0),
                    "F1 (fraud)": f1_score(y_test, pred, zero_division=0)
                }
                st.session_state["cm"] = confusion_matrix(y_test, pred)
                st.session_state["report"] = classification_report(
                    y_test, pred, target_names=["Genuine", "Fraudulent"],
                    zero_division=0
                )
                st.session_state["train_y"] = y_train
                st.session_state["test_y"] = y_test
                st.session_state["test_pred"] = pred
                st.session_state["clean_data"] = work
            st.success("Model trained successfully using an 80/20 stratified split.")

if page == "Train & Evaluate":
    st.subheader("2. Model evaluation")
    if "metrics" not in st.session_state:
        st.warning("Click **Train / refresh model** above to train the model.")
    else:
        cols = st.columns(4)
        for col, (name, value) in zip(cols, st.session_state["metrics"].items()):
            col.metric(name, f"{value:.3f}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(
                st.session_state["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Genuine", "Fraudulent"],
                yticklabels=["Genuine", "Fraudulent"], ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.markdown("**Classification report**")
            st.code(st.session_state["report"])
        st.caption("Metrics are calculated from the held-out test set. TF-IDF is fitted only on training text.")

if page == "Check a Job":
    st.subheader("3. Screen a job advertisement")
    if "model" not in st.session_state:
        st.warning("Train the model first using the button above.")
    else:
        title = st.text_input("Job title", placeholder="e.g. Data Analyst Intern")
        company = st.text_area("Company profile (optional)", height=80)
        description = st.text_area(
            "Job description", height=180,
            placeholder="Paste the job advertisement here..."
        )
        requirements = st.text_area("Requirements (optional)", height=80)
        benefits = st.text_area("Benefits (optional)", height=80)
        if st.button("Analyse advertisement", type="primary"):
            combined = " ".join([title, company, description, requirements, benefits])
            if not combined.strip():
                st.warning("Enter a job title or description first.")
            else:
                vector = st.session_state["vectorizer"].transform([combined])
                model = st.session_state["model"]
                prediction = int(model.predict(vector)[0])
                fraud_score = float(model.predict_proba(vector)[0][1])
                if prediction == 1:
                    st.error("Model result: Potentially fraudulent — needs further verification.")
                else:
                    st.success("Model result: Not flagged as fraudulent by this model.")
                st.metric("Model-estimated fraud score", f"{fraud_score * 100:.1f}%")
                st.caption(
                    "This score is a model estimate, not a verified probability or proof of fraud. "
                    "Independently verify the employer and never pay upfront recruitment fees."
                )

if page == "Data Mining Insights":
    st.subheader("4. Data Mining insights")
    if "clean_data" not in st.session_state:
        st.warning("Train the model first using the button above.")
    else:
        data = st.session_state["clean_data"]
        a, b = st.columns(2)
        with a:
            st.markdown("**Class distribution**")
            counts = data["fraudulent"].value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(["Genuine (0)", "Fraudulent (1)"], [counts.get(0, 0), counts.get(1, 0)])
            ax.set_ylabel("Number of postings")
            st.pyplot(fig)
            plt.close(fig)
        with b:
            st.markdown("**Missing values by column (top 15)**")
            missing = data.isna().sum().sort_values(ascending=False).head(15)
            st.dataframe(missing.rename("Missing values"), use_container_width=True)

        st.markdown("**K-Means clustering on TF-IDF sample**")
        texts, _ = make_text_frame(data)
        sample_n = min(3000, len(data))
        sample_texts = texts.sample(sample_n, random_state=42)
        vec = TfidfVectorizer(max_features=500, stop_words="english", min_df=2)
        mat = vec.fit_transform(sample_texts)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(mat)
        st.write(pd.Series(labels, name="Cluster").value_counts().sort_index().rename("Records"))

        st.markdown("**Apriori association rules**")
        needed = ["company_profile", "has_company_logo", "telecommuting", "has_questions", "salary_range", "fraudulent"]
        if all(c in data.columns for c in needed):
            basket = pd.DataFrame({
                "No_Company_Profile": data["company_profile"].fillna("").eq("").astype(bool),
                "No_Company_Logo": pd.to_numeric(data["has_company_logo"], errors="coerce").fillna(0).eq(0),
                "Remote_Job": pd.to_numeric(data["telecommuting"], errors="coerce").fillna(0).eq(1),
                "Has_Questions": pd.to_numeric(data["has_questions"], errors="coerce").fillna(0).eq(1),
                "No_Salary_Info": data["salary_range"].fillna("").eq("").astype(bool),
                "Fraudulent": data["fraudulent"].eq(1)
            })
            itemsets = apriori(basket, min_support=0.01, use_colnames=True)
            if len(itemsets):
                rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
                st.dataframe(rules.head(20), use_container_width=True)
                if rules.empty:
                    st.info("Frequent itemsets were found, but no rules met the selected confidence threshold.")
            else:
                st.info("No frequent itemsets met the selected support threshold.")
        else:
            st.info("Apriori needs the expected company, logo, remote-work, questions, salary and fraudulent columns.")

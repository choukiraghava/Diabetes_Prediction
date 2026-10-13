import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Diabetes AI | Health Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "diabetes.csv"
MODEL_PATH = "models/best_model.joblib"
RESULTS_PATH = "models/results.json"
FEATURES_PATH = "models/input_features.json"
PREDICTIONS_PATH = "models/test_predictions.csv"

LABELS = ["Y", "N", "P"]

# ---------- Dark theme ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1729 100%);
    color: #eef4ff;
}
[data-testid="stSidebar"] {
    background: #0b1423;
    border-right: 1px solid #24344c;
}
h1, h2, h3 {
    color: #f2f7ff;
}
p, label, .stMarkdown {
    color: #d2dced;
}
[data-testid="stMetric"] {
    background: #111f33;
    border: 1px solid #263b55;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stForm"] {
    background: #101e31;
    padding: 20px;
    border: 1px solid #293e57;
    border-radius: 15px;
}
.stButton button, .stFormSubmitButton button {
    background: #14b8a6;
    color: #07111f;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: #5eead4;
    color: #07111f;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #14243a;
}
.small-muted {
    color: #91a6c3;
    font-size: 13px;
}
.hero {
    padding: 25px;
    border: 1px solid #24445f;
    border-radius: 18px;
    background: linear-gradient(110deg, #102b43, #102137);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ---------- Load project files ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_features():
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)


missing = [
    path for path in [
        DATA_PATH, MODEL_PATH, RESULTS_PATH, FEATURES_PATH
    ] if not os.path.exists(path)
]

if missing:
    st.error(
        "Some project files are missing: "
        + ", ".join(missing)
        + ". Run `python train_model.py` first and check the project folder."
    )
    st.stop()

model = load_model()
df = load_dataset()
results = load_results()
features = load_features()

df.columns = df.columns.str.strip()
if "CLASS" in df.columns:
    df["CLASS"] = df["CLASS"].astype(str).str.strip()

# Use the actual feature columns from training.
input_df = df[features].copy()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🩺 Diabetes AI")
    st.caption("Machine Learning Health Analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Class",
            "Model Analysis",
            "Dataset Explorer",
        ],
    )

    st.divider()
    st.markdown("### Project Information")
    st.caption("Dataset: Diabetes classification")
    st.caption(f"Records: {len(df):,}")
    st.caption(f"Model: {results['selected_model']}")
    st.markdown(
        '<p class="small-muted">Educational project — not a medical diagnosis.</p>',
        unsafe_allow_html=True,
    )


# ---------- Dashboard ----------
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>Diabetes Prediction Dashboard</h1>
            <p>
            Explore healthcare data, understand model performance,
            and generate a machine-learning class prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Input Features", len(features))
    c3.metric(
        "Test Accuracy",
        f"{results['test_accuracy'] * 100:.2f}%"
    )
    c4.metric(
        "Test Macro F1",
        f"{results['test_f1_macro']:.3f}"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Target Class Distribution")

        if "CLASS" in df.columns:
            counts = (
                df["CLASS"].value_counts()
                .reindex(LABELS, fill_value=0)
                .rename_axis("Class")
                .reset_index(name="Records")
            )

            fig = px.bar(
                counts,
                x="Class",
                y="Records",
                color="Class",
                color_discrete_sequence=[
                    "#14b8a6", "#60a5fa", "#c084fc"
                ],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model Performance")

        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "Macro F1"],
            "Score": [
                results["test_accuracy"],
                results["test_precision_macro"],
                results["test_recall_macro"],
                results["test_f1_macro"],
            ],
        })

        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            range_y=[0, 1],
            color="Metric",
            color_discrete_sequence=[
                "#14b8a6", "#60a5fa", "#c084fc", "#fbbf24"
            ],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The model predicts a dataset class label (Y, N, or P). "
        "Confirm the original dataset documentation for the exact "
        "meaning of these labels."
    )


# ---------- Prediction ----------
elif page == "Predict Class":
    st.title("Patient Feature Assessment")
    st.write(
        "Enter the feature values to obtain a prediction from the trained model."
    )

    st.warning(
        "For educational use only. This result is not a medical diagnosis "
        "and should not guide treatment decisions."
    )

    numeric_cols = input_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = [
        col for col in features if col not in numeric_cols
    ]

    with st.form("prediction_form"):
        st.subheader("Medical and Demographic Features")

        values = {}
        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):
            container = col1 if i % 2 == 0 else col2
            series = input_df[feature]

            with container:
                if feature in categorical_cols:
                    options = sorted(
                        series.dropna().astype(str).unique().tolist()
                    )
                    if not options:
                        options = ["Unknown"]

                    values[feature] = st.selectbox(
                        feature, options=options
                    )

                else:
                    clean = pd.to_numeric(
                        series, errors="coerce"
                    ).dropna()

                    if clean.empty:
                        default = 0.0
                        minimum = 0.0
                        maximum = 100.0
                    else:
                        default = float(clean.median())
                        minimum = float(clean.min())
                        maximum = float(clean.max())

                    if minimum == maximum:
                        minimum -= 1
                        maximum += 1

                    values[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        step=0.1,
                        help="Initial limits are based on the dataset.",
                    )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True,
        )

    if submitted:
        try:
            patient = pd.DataFrame(
                [[values[f] for f in features]],
                columns=features,
            )

            for col in numeric_cols:
                patient[col] = pd.to_numeric(
                    patient[col], errors="raise"
                )

            prediction = str(model.predict(patient)[0])

            st.divider()
            st.subheader("Prediction Result")

            st.metric("Predicted Dataset Class", prediction)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(patient)[0]
                class_names = model.named_steps[
                    "classifier"
                ].classes_

                prob_df = pd.DataFrame({
                    "Class": class_names,
                    "Probability": probabilities,
                })

                fig = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    range_y=[0, 1],
                    color="Class",
                    template="plotly_dark",
                    title="Model-Estimated Class Probabilities",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Y, N, and P are dataset labels. Their exact clinical "
                "interpretation must be verified from the dataset source."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------- Model Analysis ----------
elif page == "Model Analysis":
    st.title("Model Evaluation & Analysis")
    st.write(
        f"Selected model: **{results['selected_model']}**"
    )

    metrics = [
        ("Accuracy", results["test_accuracy"]),
        ("Macro Precision", results["test_precision_macro"]),
        ("Macro Recall", results["test_recall_macro"]),
        ("Macro F1-score", results["test_f1_macro"]),
    ]

    cols = st.columns(4)
    for col, (name, score) in zip(cols, metrics):
        col.metric(name, f"{score:.4f}")

    st.subheader("Confusion Matrix")

    cm = np.array(results["confusion_matrix"])

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {x}" for x in LABELS],
            y=[f"Actual {x}" for x in LABELS],
            text=cm,
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=True,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")

    report = pd.DataFrame(
        results["classification_report"]
    ).T

    st.dataframe(
        report.round(4),
        use_container_width=True,
    )

    st.subheader("Feature Importance")

    if os.path.exists(PREDICTIONS_PATH):
        test_data = pd.read_csv(PREDICTIONS_PATH)

        if "Actual_Class" in test_data.columns:
            test_X = test_data[features]
            test_y = test_data["Actual_Class"]

            with st.spinner("Calculating feature importance..."):
                try:
                    importance_result = permutation_importance(
                        model,
                        test_X,
                        test_y,
                        scoring="f1_macro",
                        n_repeats=5,
                        random_state=42,
                    )

                    importance = pd.DataFrame({
                        "Feature": features,
                        "Importance": importance_result.importances_mean,
                    }).sort_values("Importance", ascending=False)

                    fig = px.bar(
                        importance,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        template="plotly_dark",
                        title="Permutation Feature Importance",
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(
                        fig, use_container_width=True
                    )
                    st.caption(
                        "Importance is measured by the change in "
                        "macro F1 when a feature is shuffled. Negative "
                        "values can occur and do not imply causation."
                    )

                except Exception as e:
                    st.warning(f"Feature importance unavailable: {e}")
    else:
        st.info(
            "Run train_model.py to generate test predictions "
            "for feature-importance analysis."
        )


# ---------- Dataset Explorer ----------
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Cells", int(df.isna().sum().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Feature Summary")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True,
    )

    st.subheader("Missing Values")
    missing_df = (
        df.isna().sum()
        .rename("Missing Values")
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    fig = px.bar(
        missing_df,
        x="Feature",
        y="Missing Values",
        template="plotly_dark",
        title="Missing Values by Feature",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Dataset Preview",
        data=df.head(100).to_csv(index=False).encode("utf-8"),
        file_name="diabetes_dataset_preview.csv",
        mime="text/csv",
    )


# ---------- Footer ----------
st.divider()
st.markdown(
    """
    <p class="small-muted" style="text-align:center;">
    Diabetes AI · Machine Learning Academic Project ·
    Educational use only, not a medical diagnosis
    </p>
    """,
    unsafe_allow_html=True,
)
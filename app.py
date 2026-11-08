
import os
import json
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)


# =====================================================
# 2. CUSTOM STYLING - STREAMLIT THEME AWARE
# =====================================================

st.markdown("""
<style>
/* LIGHT MODE: high-contrast dark text on light surfaces */
:root {
    --app-bg: #f5f7ff;
    --app-text: #172b4d;
    --heading-color: #142b45;
    --card-bg: #ffffff;
    --metric-bg: #e8f7f2;
    --metric-text: #075c45;
    --border-color: #b7dfd2;
    --muted-text: #425466;
    color-scheme: light;
}

.stApp { background: var(--app-bg) !important; color: var(--app-text) !important; }
.stApp p, .stApp li, .stApp label, .stApp small,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] * {
    color: var(--app-text) !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--heading-color) !important;
}

/* Keep the colorful title */
.main-title {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg, #11998e, #38ef7d);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.3;
}
.sub-title { color: var(--muted-text) !important; font-size: 16px; font-weight: 500; }

/* Student card and sidebar intentionally remain dark with white text */
.student-card {
    background: linear-gradient(135deg, #243b55, #141e30);
    padding: 15px 20px; border-radius: 14px; text-align: right;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}
.student-card p, .student-card span, .student-card .student-name {
    color: #ffffff !important; margin: 4px 0;
}
.student-card p { font-size: 14px; }
.student-card .student-name { font-size: 17px; font-weight: 700; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #162b4d, #285b70) !important;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--metric-bg) !important;
    padding: 20px; border-radius: 15px;
    border: 1px solid var(--border-color);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {
    color: var(--app-text) !important; font-weight: 700;
}
div[data-testid="stMetricValue"] *, div[data-testid="stMetricValue"] {
    color: var(--metric-text) !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #11998e, #38ef7d) !important;
    color: #ffffff !important; border: none; border-radius: 10px;
    font-weight: 700; padding: 10px 20px; transition: all 0.2s ease;
}
.stButton > button *, .stDownloadButton > button * { color: #ffffff !important; }
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(90deg, #087f5b, #11998e) !important;
    transform: translateY(-2px);
}

/* Uploader, dropdowns and input controls */
div[data-testid="stFileUploader"] {
    background: var(--card-bg) !important; border: 2px dashed #38a890;
    border-radius: 15px; padding: 15px;
}
div[data-testid="stFileUploader"] *,
div[data-testid="stFileUploader"] button * {
    color: var(--app-text) !important;
}
div[data-testid="stFileUploader"] button {
    background: var(--card-bg) !important; border: 1px solid #38a890 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: var(--card-bg) !important;
}
div[data-baseweb="select"] *, div[data-baseweb="popover"] *,
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    color: var(--app-text) !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--card-bg) !important;
}

/* Expanders, alerts and separators */
div[data-testid="stExpander"] {
    background: var(--card-bg) !important; border: 1px solid var(--border-color);
    border-radius: 12px;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary * { color: var(--app-text) !important; }
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
    color: var(--app-text) !important;
}
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
hr { border-color: var(--border-color) !important; }

/* DARK MODE: restore the previous dark appearance without changing it */
@media (prefers-color-scheme: dark) {
    :root {
        --app-bg: #101827;
        --app-text: #e5e7eb;
        --heading-color: #f1f5f9;
        --card-bg: #1e293b;
        --metric-bg: #243b45;
        --metric-text: #6ee7b7;
        --border-color: #3b5261;
        --muted-text: #cbd5e1;
        color-scheme: dark;
    }
    .stApp { background: var(--app-bg) !important; color: var(--app-text) !important; }
    .stApp p, .stApp li, .stApp label, .stApp small,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] *,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] *,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] * {
        color: var(--app-text) !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--heading-color) !important;
    }
    div[data-testid="stMetric"] { background: var(--metric-bg) !important; }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {
        color: var(--app-text) !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        color: var(--metric-text) !important;
    }
    div[data-testid="stFileUploader"],
    div[data-testid="stExpander"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: var(--card-bg) !important;
    }
    div[data-testid="stFileUploader"] *,
    div[data-testid="stFileUploader"] button *,
    div[data-baseweb="select"] *, div[data-baseweb="popover"] *,
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: var(--app-text) !important;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: var(--card-bg) !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
        color: var(--app-text) !important;
    }
    hr { border-color: var(--border-color) !important; }
}


/* =====================================================
   FINAL OVERRIDE: ORANGE TYPOGRAPHY IN BOTH THEMES
   Background colors above are intentionally untouched.
===================================================== */
.stApp,
.stApp *,
.stApp *::before,
.stApp *::after,
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
div[data-testid="stMetric"] *,
div[data-testid="stFileUploader"] *,
div[data-testid="stExpander"] *,
div[data-testid="stAlert"] *,
div[data-testid="stDataFrame"] *,
div[data-testid="stTable"] *,
div[data-baseweb="select"] *,
div[data-baseweb="popover"] *,
.stButton button,
.stButton button *,
.stDownloadButton button,
.stDownloadButton button *,
input,
textarea,
label,
small,
p,
span,
li,
h1, h2, h3, h4, h5, h6 {
    color: #FF8C00 !important;
}

/* Keep the title solid orange instead of gradient-clipped text. */
.main-title {
    color: #FF8C00 !important;
    background: none !important;
    -webkit-text-fill-color: #FF8C00 !important;
}
.sub-title {
    color: #FF8C00 !important;
}

/* Ensure input text and placeholder text are orange. */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    color: #FF8C00 !important;
    -webkit-text-fill-color: #FF8C00 !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #FF8C00 !important;
    opacity: 0.8 !important;
}

/* Student card and sidebar typography also use orange. */
.student-card,
.student-card *,
section[data-testid="stSidebar"] * {
    color: #FF8C00 !important;
}

/* Metric labels, values and deltas. */
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"] * {
    color: #FF8C00 !important;
}

/* Keep button backgrounds as originally styled; only text changes. */
.stButton > button,
.stDownloadButton > button,
.stButton > button *,
.stDownloadButton > button * {
    color: #FF8C00 !important;
}

/* SVG icons inherit the same orange foreground where supported. */
.stApp svg {
    color: #FF8C00 !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HEADER
# =====================================================

header_col1, header_col2 = st.columns([3, 1.5])

with header_col1:

    st.markdown(
        '<div class="main-title">'
        '🍄 Mushroom Classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Machine Learning Model Evaluation Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

with header_col2:

    st.markdown("""
    <div class="student-card">
        <p class="student-name">👨‍🎓 Vikneshkumar M</p>
        <p>BITS ID: 2026AJ05896</p>
        <p>M.Tech AIML</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

st.warning(
    "Academic demonstration only. Never use this application "
    "to determine whether a real mushroom is safe to consume."
)


# =====================================================
# 4. LOAD METADATA
# =====================================================

METADATA_PATH = "models/metadata.json"

if not os.path.exists(METADATA_PATH):
    st.error(
        "Model metadata not found. "
        "Please run train_models.py first."
    )
    st.stop()

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

feature_columns = metadata.get("features", [])
target_column = metadata.get("target_column", "class")

if not feature_columns:
    st.error(
        "Feature information missing in metadata.json. "
        "Please run train_models.py again."
    )
    st.stop()


# =====================================================
# 5. MODEL OPTIONS
# =====================================================

model_options = {
    "Logistic Regression":
        "logistic_regression.joblib",

    "Decision Tree":
        "decision_tree.joblib",

    "K-Nearest Neighbors":
        "knn.joblib",

    "Naive Bayes":
        "naive_bayes.joblib",

    "Support Vector Machine":
        "support_vector_machine.joblib"
}


# =====================================================
# 6. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Configuration")

selected_model = st.sidebar.selectbox(
    "Select ML Model",
    list(model_options.keys())
)

st.sidebar.divider()

st.sidebar.markdown("""
### Project Information

**Dataset:** UCI Mushroom

**Features:** 22

**Target:** Edible / Poisonous

**Algorithms:** 5

**Metrics:** 6
""")


# =====================================================
# 7. FILE UPLOAD
# =====================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload the test_data.csv generated by "
        "train_models.py to evaluate the selected model."
    )

    st.caption(
        "The CSV must contain all 22 features and "
        "the actual class target column."
    )

    st.stop()


# =====================================================
# 8. READ AND VALIDATE CSV
# =====================================================

try:

    uploaded_data = pd.read_csv(uploaded_file)

    if uploaded_data.empty:
        st.error("Uploaded CSV is empty.")
        st.stop()

    if target_column not in uploaded_data.columns:
        st.error(
            f"Target column '{target_column}' is missing."
        )
        st.stop()

    missing_features = [
        column
        for column in feature_columns
        if column not in uploaded_data.columns
    ]

    if missing_features:
        st.error(
            "Missing required features: "
            + ", ".join(missing_features)
        )
        st.stop()

    X_test = uploaded_data[feature_columns].copy()

    y_test = (
        uploaded_data[target_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "e": "edible",
            "p": "poisonous",
            "edible": "edible",
            "poisonous": "poisonous"
        })
    )

    if y_test.isna().any():
        st.error(
            "Invalid or missing target values. "
            "Use edible/poisonous or e/p."
        )
        st.stop()

    if y_test.nunique() != 2:
        st.error(
            "Both edible and poisonous classes must be "
            "present to calculate all evaluation metrics."
        )
        st.stop()

except Exception as error:

    st.error(f"CSV processing error: {error}")
    st.stop()


# =====================================================
# 9. DISPLAY DATASET INFORMATION
# =====================================================

st.header("2. Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Test Records",
    len(uploaded_data)
)

col2.metric(
    "Input Features",
    len(feature_columns)
)

col3.metric(
    "Target Classes",
    y_test.nunique()
)

with st.expander("Preview uploaded dataset"):

    st.dataframe(
        uploaded_data.head(10),
        use_container_width=True
    )


# =====================================================
# 10. LOAD SELECTED MODEL
# =====================================================

model_path = os.path.join(
    "models",
    model_options[selected_model]
)

if not os.path.exists(model_path):
    st.error(
        f"Saved model not found: {model_path}"
    )
    st.stop()

model = joblib.load(model_path)


# =====================================================
# 11. PREDICTION
# =====================================================

try:

    predictions = model.predict(X_test)

    classes = list(model.classes_)

    poisonous_index = classes.index("poisonous")

    probabilities = model.predict_proba(X_test)[
        :, poisonous_index
    ]

except Exception as error:

    st.error(f"Prediction failed: {error}")
    st.stop()


# =====================================================
# 12. CALCULATE METRICS
# =====================================================

y_binary = (
    y_test == "poisonous"
).astype(int)

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_binary,
    probabilities
)

precision = precision_score(
    y_test,
    predictions,
    pos_label="poisonous",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="poisonous",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="poisonous",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    predictions
)


# =====================================================
# 13. DISPLAY EVALUATION METRICS
# =====================================================

st.divider()

st.header("3. Model Evaluation")

st.write(
    f"**Selected Model:** {selected_model}"
)

metric_row1 = st.columns(3)

metric_row1[0].metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

metric_row1[1].metric(
    "AUC Score",
    f"{auc:.4f}"
)

metric_row1[2].metric(
    "Precision",
    f"{precision:.4f}"
)

metric_row2 = st.columns(3)

metric_row2[0].metric(
    "Recall",
    f"{recall:.4f}"
)

metric_row2[1].metric(
    "F1 Score",
    f"{f1:.4f}"
)

metric_row2[2].metric(
    "MCC Score",
    f"{mcc:.4f}"
)


# =====================================================
# 14. CONFUSION MATRIX
# =====================================================

st.divider()

st.header("4. Confusion Matrix")

labels = ["edible", "poisonous"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("Mushroom Classification Confusion Matrix")

st.pyplot(fig)

plt.close(fig)


# =====================================================
# 15. CLASSIFICATION REPORT
# =====================================================

st.divider()

st.header("5. Classification Report")

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# =====================================================
# 16. PREDICTION RESULTS
# =====================================================

st.divider()

st.header("6. Prediction Results")

result_df = X_test.copy()

result_df["Actual Class"] = y_test.values

result_df["Predicted Class"] = predictions

result_df["Prediction Status"] = [
    "Correct" if actual == predicted else "Incorrect"
    for actual, predicted in zip(y_test, predictions)
]

st.dataframe(
    result_df.head(20),
    use_container_width=True
)

csv_data = result_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=csv_data,
    file_name="mushroom_predictions.csv",
    mime="text/csv"
)


# =====================================================
# 17. COMPLETION
# =====================================================

st.divider()

st.success(
    "Model evaluation completed successfully!"
)

st.caption(
    "M.Tech AIML | Machine Learning Assignment 2"
)

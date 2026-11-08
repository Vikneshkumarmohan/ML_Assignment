
import os
import json
import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ------------------------------------------------
# 1. Configuration
# ------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "class"

os.makedirs("models", exist_ok=True)

print("Downloading Mushroom dataset from UCI...")

# ------------------------------------------------
# 2. Download dataset
# ------------------------------------------------

mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features.copy()
y = mushroom.data.targets.iloc[:, 0].copy()

X.columns = [
    str(column).strip().lower().replace("-", "_")
    for column in X.columns
]

# Handle original missing-value marker
X = X.replace("?", np.nan)

# Convert original target labels
y = y.astype(str).str.strip().str.lower()

y = y.map({
    "e": "edible",
    "p": "poisonous",
    "edible": "edible",
    "poisonous": "poisonous"
})

if y.isna().any():
    raise ValueError("Unexpected target labels found in dataset.")

y.name = TARGET

data = X.copy()
data[TARGET] = y

print("Dataset downloaded successfully.")
print("Dataset shape:", data.shape)
print("Missing values:", int(X.isna().sum().sum()))

# ------------------------------------------------
# 3. Train-test split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Save held-out test dataset.
# This contains all 22 features and the actual target.
test_data = X_test.copy()
test_data[TARGET] = y_test

test_data.to_csv("test_data.csv", index=False)

print("test_data.csv generated successfully.")

# ------------------------------------------------
# 4. Preprocessing pipeline
# ------------------------------------------------

preprocessor = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

# ------------------------------------------------
# 5. Define five classifiers
# ------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=15
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7
    ),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    )
}

# ------------------------------------------------
# 6. Train and evaluate
# ------------------------------------------------

results = []

for model_name, classifier in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train using training data only
    pipeline.fit(X_train, y_train)

    # Predict test data
    predictions = pipeline.predict(X_test)

    # Get probability for poisonous class
    classes = list(pipeline.classes_)
    poisonous_index = classes.index("poisonous")

    probabilities = pipeline.predict_proba(X_test)[
        :, poisonous_index
    ]

    # Calculate evaluation metrics
    metrics = {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "AUC": roc_auc_score(
            (y_test == "poisonous").astype(int),
            probabilities
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            pos_label="poisonous",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            pos_label="poisonous",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            pos_label="poisonous",
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, predictions
        )
    }

    results.append(metrics)

    # Save complete pipeline including preprocessing
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        pipeline,
        os.path.join("models", filename)
    )

    print("Saved:", filename)

# ------------------------------------------------
# 7. Save evaluation results
# ------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "metrics_summary.csv",
    index=False
)

print("\nMODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

# ------------------------------------------------
# 8. Save project metadata
# ------------------------------------------------

metadata = {
    "target": TARGET,
    "features": list(X.columns),
    "classes": list(y.unique()),
    "positive_class": "poisonous",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE
}

with open("models/metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("\nAll models trained successfully.")
print("Test dataset and metrics saved.")
print("Project training completed.")

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "diabetes.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

if "CLASS" not in df.columns:
    raise ValueError("Dataset must contain a CLASS target column.")

df["CLASS"] = df["CLASS"].astype(str).str.strip()
df = df.dropna(subset=["CLASS"])

# Remove blank or invalid target labels.
df = df[df["CLASS"].isin(["Y", "N", "P"])].copy()

# 2. Separate features and target
X = df.drop(columns=["CLASS", "ID", "No_Pation"], errors="ignore")
y = df["CLASS"]

if X.empty:
    raise ValueError("No input features found after removing identifier columns.")

# Detect feature types
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_cols = [
    col for col in X.columns if col not in categorical_cols
]

# 3. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# 4. Split into train, validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.1765,
    random_state=42,
    stratify=y_train_val,
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    ),
}

# 5. Train and compare using validation macro F1
validation_results = {}
trained_models = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    score = f1_score(
        y_val, predictions,
        labels=["Y", "N", "P"],
        average="macro",
        zero_division=0,
    )

    validation_results[name] = float(score)
    trained_models[name] = pipeline

    print(f"{name} validation macro F1: {score:.4f}")

best_name = max(validation_results, key=validation_results.get)
best_model = trained_models[best_name]

print("\nSelected model:", best_name)

# 6. Refit selected model on training + validation data
best_model.fit(X_train_val, y_train_val)

# 7. Final evaluation on held-out test data
y_pred = best_model.predict(X_test)
labels = ["Y", "N", "P"]

metrics = {
    "selected_model": best_name,
    "validation_macro_f1": validation_results,
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_precision_macro": float(
        precision_score(y_test, y_pred, average="macro",
                        zero_division=0)
    ),
    "test_recall_macro": float(
        recall_score(y_test, y_pred, average="macro",
                     zero_division=0)
    ),
    "test_f1_macro": float(
        f1_score(y_test, y_pred, average="macro",
                 zero_division=0)
    ),
    "class_labels": labels,
    "test_records": int(len(y_test)),
}

cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\nTest results:")
print(json.dumps(metrics, indent=4))
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in labels],
    columns=[f"Predicted {c}" for c in labels],
))
print("\nClassification report:")
print(classification_report(
    y_test, y_pred, labels=labels, zero_division=0
))

# 8. Save model and results
joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")

results = {
    **metrics,
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_test, y_pred, labels=labels,
        output_dict=True, zero_division=0
    ),
}

with open(f"{MODEL_DIR}/results.json", "w") as file:
    json.dump(results, file, indent=4)

# Save test predictions for later analysis
test_output = X_test.copy()
test_output["Actual_Class"] = y_test
test_output["Predicted_Class"] = y_pred
test_output.to_csv(f"{MODEL_DIR}/test_predictions.csv", index=False)

# Save input feature names
with open(f"{MODEL_DIR}/input_features.json", "w") as file:
    json.dump(list(X.columns), file, indent=4)

print("\nModel and evaluation results saved in the models folder.")
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = BASE_DIR / "dataset" / "credit_risk_dataset.csv"
DEFAULT_MODEL_OUTPUT_PATH = BASE_DIR / "models" / "credit_risk_pipeline.joblib"
DATASET_PATH = Path(os.getenv("CREDIT_RISK_DATASET_PATH", DEFAULT_DATASET_PATH))
MODEL_OUTPUT_PATH = Path(os.getenv("CREDIT_RISK_MODEL_PATH", DEFAULT_MODEL_OUTPUT_PATH))
DEFAULT_DASHBOARD_PATH = BASE_DIR / "credit_risk_dashboard.png"
TARGET_COLUMN = "loan_status"
FEATURE_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]
SPLIT_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 200
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
THRESHOLD_BETA = 2.0


def clean_credit_data(df):
    df = df.copy().drop_duplicates()

    numeric_columns = [
        "person_age",
        "person_emp_length",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_emp_length = df["person_emp_length"].isna() | (
        df["person_emp_length"] <= df["person_age"] - 15
    )

    removed_rows = int((~valid_emp_length).sum())
    df = df.loc[valid_emp_length].copy()

    return df, removed_rows


def build_pipeline(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=MODEL_RANDOM_STATE,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


def predict_from_probabilities(probabilities, threshold):
    return (probabilities >= threshold).astype(int)


def choose_decision_threshold(y_true, probabilities, beta=THRESHOLD_BETA):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    scores = (
        (1 + beta**2)
        * precision[:-1]
        * recall[:-1]
        / ((beta**2) * precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.nanargmax(scores))

    return {
        "threshold": float(thresholds[best_index]),
        "fbeta_score": float(scores[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "beta": float(beta),
    }


def visualise(
    y_test,
    y_pred,
    X_test,
    model,
    threshold,
    save_path=DEFAULT_DASHBOARD_PATH,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=1.1)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0], cbar=False)
    ax[0].set_title("Confusion Matrix")
    ax[0].set_xlabel("Predicted Status")
    ax[0].set_ylabel("Actual Status")

    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocess.get_feature_names_out()
    importances = pd.Series(
        classifier.feature_importances_, index=feature_names
    ).sort_values(ascending=True)
    importances.tail(10).plot(kind="barh", ax=ax[1], color="#2ca02c")
    ax[1].set_title("Top 10 Drivers of Credit Risk")
    ax[1].set_xlabel("Importance Score")

    PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, ax=ax[2], color="#d62728"
    )
    ax[2].set_title("Precision-Recall Curve")

    plt.suptitle(
        f"Credit Risk Model Evaluation (threshold={threshold:.3f})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_artifact(
    model,
    threshold,
    tuning_summary,
    output_path=MODEL_OUTPUT_PATH,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "decision_threshold": float(threshold),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "threshold_strategy": "validation_fbeta",
        "threshold_beta": float(tuning_summary["beta"]),
        "threshold_validation_precision": float(tuning_summary["precision"]),
        "threshold_validation_recall": float(tuning_summary["recall"]),
        "threshold_validation_fbeta": float(tuning_summary["fbeta_score"]),
    }
    joblib.dump(artifact, output_path)
    return output_path


def load_model_artifact(model_path=MODEL_OUTPUT_PATH):
    artifact = joblib.load(model_path)

    if hasattr(artifact, "predict_proba"):
        return {
            "model": artifact,
            "decision_threshold": 0.5,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "threshold_strategy": "default_probability_cutoff",
            "threshold_beta": None,
            "threshold_validation_precision": None,
            "threshold_validation_recall": None,
            "threshold_validation_fbeta": None,
        }

    return artifact


def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Rows before cleaning: {len(df)}")
    df, removed_rows = clean_credit_data(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y_train_full,
    )

    tuning_model = build_pipeline(X_train)
    tuning_model.fit(X_train, y_train)

    val_probabilities = tuning_model.predict_proba(X_val)[:, 1]
    tuning_summary = choose_decision_threshold(y_val, val_probabilities)
    tuned_threshold = tuning_summary["threshold"]

    model = build_pipeline(X_train_full)
    model.fit(X_train_full, y_train_full)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = predict_from_probabilities(y_prob, threshold=0.5)
    y_pred_tuned = predict_from_probabilities(y_prob, threshold=tuned_threshold)

    print(f"Rows after cleaning: {len(df)}")
    print(f"Rows removed by employment-length rule: {removed_rows}")
    print(f"Train shape: {X_train_full.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Average precision: {average_precision_score(y_test, y_prob):.4f}")
    print(
        f"Tuned threshold from validation (F{THRESHOLD_BETA:.0f}): "
        f"{tuned_threshold:.4f}"
    )
    print(
        "Validation threshold summary: "
        f"precision={tuning_summary['precision']:.4f}, "
        f"recall={tuning_summary['recall']:.4f}, "
        f"F{THRESHOLD_BETA:.0f}={tuning_summary['fbeta_score']:.4f}"
    )
    print(
        "Default threshold (0.5000): "
        f"accuracy={accuracy_score(y_test, y_pred_default):.4f}, "
        f"risk_precision={precision_score(y_test, y_pred_default):.4f}, "
        f"risk_recall={recall_score(y_test, y_pred_default):.4f}"
    )
    print(
        f"Tuned threshold ({tuned_threshold:.4f}): "
        f"accuracy={accuracy_score(y_test, y_pred_tuned):.4f}, "
        f"risk_precision={precision_score(y_test, y_pred_tuned):.4f}, "
        f"risk_recall={recall_score(y_test, y_pred_tuned):.4f}"
    )
    print(
        "Confusion matrix at tuned threshold:\n"
        f"{confusion_matrix(y_test, y_pred_tuned)}"
    )
    print("Classification report at tuned threshold:")
    print(classification_report(y_test, y_pred_tuned, digits=4))

    visualise(y_test, y_pred_tuned, X_test, model, tuned_threshold)
    saved_model_path = save_model_artifact(
        model,
        threshold=tuned_threshold,
        tuning_summary=tuning_summary,
    )
    print(f"Saved evaluation dashboard to {DEFAULT_DASHBOARD_PATH}")
    print(f"Saved trained model to {saved_model_path}")


if __name__ == "__main__":
    main()

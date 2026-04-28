import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from predict_credit_risk.training import (
    FEATURE_COLUMNS,
    MODEL_OUTPUT_PATH,
    clean_credit_data,
    load_model_artifact,
    predict_from_probabilities,
)


DEFAULT_MODEL_PATH = MODEL_OUTPUT_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the saved credit-risk model and score a CSV file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file to score.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the saved joblib model.",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to save the predictions CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual threshold override. Defaults to the saved tuned threshold.",
    )
    parser.add_argument(
        "--plot",
        default="prediction_report.png",
        help="Path to save the prediction plot image.",
    )
    return parser.parse_args()


def save_prediction_plot(results, threshold, output_path):
    has_labels = "loan_status" in results.columns
    fig, axes = plt.subplots(
        1, 3 if has_labels else 2, figsize=(18 if has_labels else 12, 5)
    )

    risk_counts = results["risk_label"].value_counts().reindex(
        ["low_risk", "high_risk"], fill_value=0
    )
    sns.barplot(
        x=risk_counts.index,
        y=risk_counts.values,
        hue=risk_counts.index,
        palette=["#2ca02c", "#d62728"],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Predicted Risk Labels")
    axes[0].set_xlabel("Risk Label")
    axes[0].set_ylabel("Count")

    sns.histplot(
        results["predicted_risk_probability"],
        bins=25,
        kde=True,
        color="#1f77b4",
        ax=axes[1],
    )
    axes[1].axvline(
        threshold, color="#d62728", linestyle="--", label=f"threshold={threshold:.3f}"
    )
    axes[1].set_title("Predicted Risk Probability")
    axes[1].set_xlabel("Risk Probability")
    axes[1].legend()

    if has_labels:
        cm = confusion_matrix(
            results["loan_status"], results["predicted_loan_status"]
        )
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2])
        axes[2].set_title("Confusion Matrix")
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    plot_path = Path(args.plot)

    artifact = load_model_artifact(model_path)
    model = artifact["model"]
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(artifact["decision_threshold"])
    )
    df = pd.read_csv(input_path)

    original_rows = len(df)
    df, removed_rows = clean_credit_data(df)

    features = df.reindex(columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = predict_from_probabilities(probabilities, threshold)

    results = df.copy()
    results["predicted_loan_status"] = predictions
    results["predicted_risk_probability"] = probabilities
    results["decision_threshold"] = threshold
    results["risk_label"] = results["predicted_loan_status"].map(
        {0: "low_risk", 1: "high_risk"}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_plot(results, threshold, plot_path)

    print(f"Loaded model from {model_path}")
    print(f"Using decision threshold: {threshold:.4f}")
    print(f"Rows removed during cleaning: {removed_rows}")
    print(f"Rows before cleaning: {original_rows}")
    print(f"Scored {len(results)} rows from {input_path}")
    print(f"Saved predictions to {output_path}")
    print(f"Saved prediction plot to {plot_path}")

    if "loan_status" in df.columns:
        print(f"Accuracy: {accuracy_score(df['loan_status'], predictions):.4f}")
        print(f"Confusion matrix:\n{confusion_matrix(df['loan_status'], predictions)}")
        print("Classification report:")
        print(classification_report(df["loan_status"], predictions, digits=4))


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd

from predict_credit_risk.training import DATASET_PATH, clean_credit_data


def main():
    df = pd.read_csv(DATASET_PATH)
    cleaned_df, removed_rows = clean_credit_data(df)

    print(f"Dataset path: {Path(DATASET_PATH)}")
    print(f"Rows before cleaning: {len(df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Rows removed: {removed_rows}")
    print("Target distribution:")
    print(cleaned_df["loan_status"].value_counts())


if __name__ == "__main__":
    main()

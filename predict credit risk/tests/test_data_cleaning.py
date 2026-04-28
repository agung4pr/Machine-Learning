import pandas as pd

from predict_credit_risk.training import clean_credit_data


def test_clean_credit_data_removes_invalid_employment_rows():
    df = pd.DataFrame(
        {
            "person_age": [30, 22],
            "person_emp_length": [20, 4],
            "person_income": [50000, 42000],
            "loan_amnt": [10000, 8000],
            "loan_int_rate": [7.5, 10.0],
            "loan_percent_income": [0.2, 0.19],
            "cb_person_cred_hist_length": [5, 3],
            "loan_status": [0, 1],
        }
    )

    cleaned_df, removed_rows = clean_credit_data(df)

    assert removed_rows == 1
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["person_age"] == 22


def test_clean_credit_data_deduplicates_rows():
    df = pd.DataFrame(
        {
            "person_age": [35, 35],
            "person_emp_length": [5, 5],
            "person_income": [60000, 60000],
            "loan_amnt": [12000, 12000],
            "loan_int_rate": [8.1, 8.1],
            "loan_percent_income": [0.2, 0.2],
            "cb_person_cred_hist_length": [7, 7],
            "loan_status": [0, 0],
        }
    )

    cleaned_df, removed_rows = clean_credit_data(df)

    assert removed_rows == 0
    assert len(cleaned_df) == 1

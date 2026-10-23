"""
CogniStream — Data Cleaning and Validation

Input:
    dataset/raw/ai_dev_productivity.csv

Output:
    dataset/processed/cleaned_data.csv

Principles:
    - Never overwrite the raw dataset.
    - Apply consistent schema and validation rules.
    - Preserve legitimate extreme observations.
    - Record invalid values instead of silently deleting rows.
"""

from pathlib import Path

import pandas as pd

from src.schema import (
    EXPECTED_COLUMNS,
    COUNT_FEATURES,
    NON_NEGATIVE_COLUMNS,
    COGNITIVE_LOAD_MIN,
    COGNITIVE_LOAD_MAX,
    TASK_SUCCESS_VALUES,
    MAX_DAILY_HOURS,
)

from src.data_collection import (
    RAW_DATA_PATH,
    load_dataset,
)


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "cleaned_data.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cleaning_report.csv"
)


# =====================================================
# 2. NUMERIC VALIDATION
# =====================================================

def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise an error when a required column is missing."""

    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def get_invalid_mask(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Identify non-missing values that violate defined rules."""

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    invalid = values.isna() & df[column].notna()

    # Non-negative variables cannot be negative.
    if column in NON_NEGATIVE_COLUMNS:
        invalid |= values < 0

    # These variables represent hours within a day.
    if column in {
        "hours_coding",
        "sleep_hours",
        "ai_usage_hours",
    }:
        invalid |= values > MAX_DAILY_HOURS

    # Counts must be whole numbers.
    if column in COUNT_FEATURES:
        invalid |= (
            values.notna()
            & (values % 1 != 0)
        )

    # Cognitive load must remain within its defined scale.
    if column == "cognitive_load":
        invalid |= (
            (values < COGNITIVE_LOAD_MIN)
            | (values > COGNITIVE_LOAD_MAX)
        )

    # Task success is a binary target.
    if column == "task_success":
        invalid |= (
            values.notna()
            & ~values.isin(TASK_SUCCESS_VALUES)
        )

    return invalid.fillna(False)


# =====================================================
# 3. CLEAN DATASET
# =====================================================

def clean_dataset(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean a copy of the source DataFrame.

    Invalid individual values are converted to missing
    values. Rows and statistical outliers are retained.
    """

    if raw_df.empty:
        raise ValueError("The input dataset is empty.")

    validate_required_columns(raw_df)

    # Work with expected features only.
    df = raw_df[EXPECTED_COLUMNS].copy()

    # Convert all dataset features to numeric values.
    for column in EXPECTED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Apply column-specific validity rules.
    for column in EXPECTED_COLUMNS:
        invalid = get_invalid_mask(
            raw_df,
            column,
        )

        df.loc[invalid, column] = float("nan")

    # Normalize successful binary target values to integers.
    # Missing values, if introduced, remain missing.
    valid_target = df["task_success"].notna()

    df.loc[valid_target, "task_success"] = (
        df.loc[valid_target, "task_success"].astype(int)
    )

    # Do not remove exact duplicate feature rows automatically:
    # without a record ID, identical values may represent
    # separate observations.
    return df.reset_index(drop=True)


# =====================================================
# 4. BUILD CLEANING REPORT
# =====================================================

def build_cleaning_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize input missingness and newly invalid values."""

    records = []

    for column in EXPECTED_COLUMNS:
        original_missing = int(
            raw_df[column].isna().sum()
        )

        newly_invalid = int(
            (
                raw_df[column].notna()
                & cleaned_df[column].isna()
            ).sum()
        )

        records.append({
            "column": column,
            "original_missing": original_missing,
            "newly_invalid": newly_invalid,
            "final_missing": int(
                cleaned_df[column].isna().sum()
            ),
        })

    return pd.DataFrame(records)


# =====================================================
# 5. SAVE PROCESSED DATASET
# =====================================================

def run_cleaning(
    input_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH,
):
    """Run cleaning and save processed data and its report."""

    raw_df = load_dataset(input_path)

    cleaned_df = clean_dataset(raw_df)

    report = build_cleaning_report(
        raw_df,
        cleaned_df,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(
        output_path,
        index=False,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("=" * 55)
    print("COGNISTREAM — CLEANING SUMMARY")
    print("=" * 55)

    print("Original shape:", raw_df.shape)
    print("Cleaned shape:", cleaned_df.shape)
    print(
        "Rows retained:",
        f"{len(cleaned_df)}/{len(raw_df)}",
    )

    print("\nColumn-level cleaning report:")
    print(report.to_string(index=False))

    print("\nProcessed dataset:", output_path)
    print("Cleaning report:", REPORT_PATH)

    return cleaned_df


# =====================================================
# 6. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    run_cleaning()
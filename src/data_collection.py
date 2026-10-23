"""
CogniStream — Data Collection and Validation
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


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "ai_dev_productivity.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_dataset(file_path=RAW_DATA_PATH):
    """Load the raw CSV without modifying the source."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place ai_dev_productivity.csv in dataset/raw/."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset contains no records.")

    return df


# --------------------------------------------------
# VALIDATE SCHEMA
# --------------------------------------------------

def validate_schema(df):
    """Check required columns and unexpected columns."""

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    return {
        "schema_valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
    }


# --------------------------------------------------
# INSPECT DATA QUALITY
# --------------------------------------------------

def inspect_dataset(df):
    """Generate a reproducible data-quality report."""

    schema = validate_schema(df)

    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "schema": schema,
        "missing_values": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_checks": {},
    }

    if not schema["schema_valid"]:
        return report

    report["missing_values"] = {
        column: int(count)
        for column, count in df.isna().sum().items()
    }

    for column in EXPECTED_COLUMNS:
        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_numeric = (
            numeric.isna() & df[column].notna()
        )

        invalid_range = pd.Series(
            False,
            index=df.index,
        )

        if column in NON_NEGATIVE_COLUMNS:
            invalid_range = numeric < 0

        if column in ["hours_coding", "sleep_hours"]:
            invalid_range |= numeric > MAX_DAILY_HOURS

        if column == "cognitive_load":
            invalid_range = (
                (numeric < COGNITIVE_LOAD_MIN)
                | (numeric > COGNITIVE_LOAD_MAX)
            )

        if column in COUNT_FEATURES:
            invalid_range |= (
                numeric.notna()
                & (numeric % 1 != 0)
            )

        if column == "task_success":
            invalid_range = (
                numeric.notna()
                & ~numeric.isin(TASK_SUCCESS_VALUES)
            )

        report["numeric_checks"][column] = {
            "invalid_numeric_values": int(
                invalid_numeric.sum()
            ),
            "invalid_range_values": int(
                (invalid_range & numeric.notna()).sum()
            ),
        }

    return report


# --------------------------------------------------
# PRINT REPORT
# --------------------------------------------------

def print_report(report):
    """Display a readable data-quality report."""

    print("=" * 60)
    print("COGNISTREAM — DATA QUALITY REPORT")
    print("=" * 60)

    print(f"Rows: {report['row_count']}")
    print(f"Columns: {report['column_count']}")

    print("\nSchema valid:", report["schema"]["schema_valid"])
    print("Missing columns:", report["schema"]["missing_columns"])
    print("Unexpected columns:", report["schema"]["unexpected_columns"])

    print("\nMissing values:")

    for column, count in report["missing_values"].items():
        print(f"{column}: {count}")

    print("\nDuplicate rows:", report["duplicate_rows"])

    print("\nNumeric validation:")

    for column, values in report["numeric_checks"].items():
        print(
            f"{column}: "
            f"invalid numeric={values['invalid_numeric_values']}, "
            f"invalid range={values['invalid_range_values']}"
        )


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    df = load_dataset()

    report = inspect_dataset(df)

    print_report(report)

    if not report["schema"]["schema_valid"]:
        raise ValueError(
            "The CSV does not match the expected schema."
        )

    print("\nData collection completed.")


if __name__ == "__main__":
    main()
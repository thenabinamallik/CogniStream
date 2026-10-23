"""
CogniStream — Feature Engineering

Creates derived variables for developer productivity analytics.

Input:
    dataset/processed/cleaned_data.csv

Output:
    dataset/processed/featured_data.csv

Note:
    Derived indicators are descriptive. They are not validated
    measurements of cognitive load or psychological flow.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import EXPECTED_COLUMNS


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "cleaned_data.csv"
)

FEATURED_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "featured_data.csv"
)


# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived variables to a cleaned dataset.

    The input DataFrame is not modified.
    """

    if df.empty:
        raise ValueError("Cannot engineer features from empty data.")

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    features = df[EXPECTED_COLUMNS].copy()

    # -------------------------------------------------
    # A. Convert original features to numeric types
    # -------------------------------------------------

    for column in EXPECTED_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    # -------------------------------------------------
    # B. Coding-hour categories
    # -------------------------------------------------

    features["coding_hours_band"] = pd.cut(
        features["hours_coding"],
        bins=[-0.001, 4, 8, np.inf],
        labels=[
            "Light (<=4h)",
            "Moderate (>4-8h)",
            "Long (>8h)",
        ],
        right=True,
    ).astype("string")

    # -------------------------------------------------
    # C. Sleep categories
    # -------------------------------------------------

    features["sleep_band"] = pd.cut(
        features["sleep_hours"],
        bins=[-np.inf, 6, 8, np.inf],
        labels=[
            "Short (<6h)",
            "Medium (6-<8h)",
            "Long (>=8h)",
        ],
        right=False,
    ).astype("string")

    # -------------------------------------------------
    # D. Cognitive-load categories
    # -------------------------------------------------

    # These are analyst-defined descriptive bands.
    # They are not validated psychological thresholds.

    features["cognitive_load_band"] = pd.cut(
        features["cognitive_load"],
        bins=[-np.inf, 3, 6, np.inf],
        labels=[
            "Low (<=3)",
            "Moderate (>3-6)",
            "High (>6)",
        ],
        right=True,
    ).astype("string")

    # -------------------------------------------------
    # E. Distraction categories
    # -------------------------------------------------

    features["distraction_band"] = pd.cut(
        features["distractions"],
        bins=[-0.001, 2, 5, np.inf],
        labels=[
            "Low (0-2)",
            "Moderate (3-5)",
            "High (6+)",
        ],
        right=True,
    ).astype("string")

    # -------------------------------------------------
    # F. Distractions per coding hour
    # -------------------------------------------------

    # Undefined when coding hours equal zero.
    coding_hours = features["hours_coding"].where(
        features["hours_coding"] > 0
    )

    features["distractions_per_coding_hour"] = (
        features["distractions"] / coding_hours
    )

    # -------------------------------------------------
    # G. Commits per coding hour
    # -------------------------------------------------

    features["commits_per_coding_hour"] = (
        features["commits"] / coding_hours
    )

    # -------------------------------------------------
    # H. Bugs reported per commit
    # -------------------------------------------------

    # Undefined when no commits were recorded.
    commit_count = features["commits"].where(
        features["commits"] > 0
    )

    features["bugs_per_commit"] = (
        features["bugs_reported"] / commit_count
    )

    # -------------------------------------------------
    # I. AI usage relative to coding hours
    # -------------------------------------------------

    features["ai_hours_per_coding_hour"] = (
        features["ai_usage_hours"] / coding_hours
    )

    # This is a ratio of recorded hours, not proof
    # that AI use occurred only during coding.

    # -------------------------------------------------
    # J. Readable task-success label
    # -------------------------------------------------

    features["task_success_label"] = (
        features["task_success"]
        .map({
            0: "Unsuccessful",
            1: "Successful",
        })
        .astype("string")
    )

    return features


# =====================================================
# 3. LOAD, TRANSFORM AND SAVE
# =====================================================

def run_feature_engineering(
    input_path=CLEANED_DATA_PATH,
    output_path=FEATURED_DATA_PATH,
) -> pd.DataFrame:
    """Load cleaned data, create features and save the result."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}\n"
            "Run python -m src.cleaning first."
        )

    cleaned_df = pd.read_csv(input_path)

    featured_df = engineer_features(cleaned_df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    featured_df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 55)
    print("COGNISTREAM — FEATURE ENGINEERING")
    print("=" * 55)

    print("Input shape:", cleaned_df.shape)
    print("Output shape:", featured_df.shape)

    print("\nNew features:")
    print(
        [
            column
            for column in featured_df.columns
            if column not in EXPECTED_COLUMNS
        ]
    )

    print("\nOutput saved:", output_path)

    return featured_df


# =====================================================
# 4. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    run_feature_engineering()
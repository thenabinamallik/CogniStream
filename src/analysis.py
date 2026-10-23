"""
CogniStream — Statistical Analysis

Analyses:
1. Descriptive summaries by task outcome.
2. Mann–Whitney U test for cognitive load by task outcome.
3. Rank-biserial effect size for the group comparison.
4. Spearman associations with cognitive load.
5. Point-biserial associations with task success.
6. Benjamini–Hochberg adjustment for multiple association tests.

This module analyzes observational dataset relationships.
It does not establish causation or directly measure flow state.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    spearmanr,
    pointbiserialr,
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURED_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "featured_data.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTOR_COLUMNS = [
    "hours_coding",
    "coffee_intake_mg",
    "distractions",
    "sleep_hours",
    "commits",
    "bugs_reported",
    "ai_usage_hours",
]

OUTCOME_COLUMNS = [
    "cognitive_load",
    "task_success",
]


# =====================================================
# 2. VALIDATE ANALYSIS INPUT
# =====================================================

def validate_analysis_data(
    df: pd.DataFrame,
) -> None:
    """Check the columns and target values required for analysis."""

    required = PREDICTOR_COLUMNS + OUTCOME_COLUMNS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required analysis columns missing: {missing}"
        )

    if df.empty:
        raise ValueError(
            "The analysis dataset is empty."
        )

    invalid_targets = (
        df["task_success"].notna()
        & ~df["task_success"].isin([0, 1])
    )

    if invalid_targets.any():
        raise ValueError(
            "task_success must contain only 0, 1, or missing values."
        )


# =====================================================
# 3. DESCRIPTIVE SUMMARY BY TASK OUTCOME
# =====================================================

def summarize_by_success(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize cognitive load for successful and
    unsuccessful tasks.
    """

    validate_analysis_data(df)

    work = df.copy()

    work["task_outcome"] = (
        work["task_success"]
        .map({
            0: "Unsuccessful",
            1: "Successful",
        })
    )

    summary = (
        work.groupby(
            "task_outcome",
            dropna=False,
        )["cognitive_load"]
        .agg(
            observations="count",
            mean="mean",
            median="median",
            standard_deviation="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    return summary.round(3)


# =====================================================
# 4. MANN–WHITNEY U TEST
# =====================================================

def compare_cognitive_load(
    df: pd.DataFrame,
) -> dict:
    """
    Compare reported cognitive load between task outcomes.

    The rank-biserial effect size is calculated from the
    Mann–Whitney U statistic for the successful group.

    Positive effect sizes indicate that cognitive-load
    values tend to be higher in successful-task records.
    Negative effect sizes indicate the opposite tendency.
    """

    validate_analysis_data(df)

    successful = (
        df.loc[
            df["task_success"] == 1,
            "cognitive_load",
        ]
        .dropna()
        .astype(float)
    )

    unsuccessful = (
        df.loc[
            df["task_success"] == 0,
            "cognitive_load",
        ]
        .dropna()
        .astype(float)
    )

    n_success = len(successful)
    n_failure = len(unsuccessful)

    result = {
        "test": "Mann–Whitney U",
        "successful_n": n_success,
        "unsuccessful_n": n_failure,
        "successful_mean": (
            float(successful.mean())
            if n_success else None
        ),
        "unsuccessful_mean": (
            float(unsuccessful.mean())
            if n_failure else None
        ),
        "successful_median": (
            float(successful.median())
            if n_success else None
        ),
        "unsuccessful_median": (
            float(unsuccessful.median())
            if n_failure else None
        ),
        "median_difference_success_minus_unsuccessful": (
            float(
                successful.median()
                - unsuccessful.median()
            )
            if n_success and n_failure
            else None
        ),
        "u_statistic": None,
        "p_value": None,
        "rank_biserial_effect_size": None,
    }

    if n_success < 2 or n_failure < 2:
        return result

    test = mannwhitneyu(
        successful,
        unsuccessful,
        alternative="two-sided",
        method="auto",
    )

    u_statistic = float(test.statistic)

    rank_biserial = (
        2 * u_statistic / (n_success * n_failure)
    ) - 1

    result.update({
        "u_statistic": u_statistic,
        "p_value": float(test.pvalue),
        "rank_biserial_effect_size": float(
            rank_biserial
        ),
    })

    return result


# =====================================================
# 5. BENJAMINI–HOCHBERG P-VALUE ADJUSTMENT
# =====================================================

def benjamini_hochberg(
    p_values,
) -> np.ndarray:
    """
    Adjust p-values using the Benjamini–Hochberg procedure.

    Missing or non-finite p-values remain missing.
    """

    p = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(p)
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p[valid_indices]

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    count = len(sorted_p)

    adjusted_sorted = (
        sorted_p
        * count
        / np.arange(1, count + 1)
    )

    # Enforce monotonic adjusted p-values.
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted_valid = np.empty_like(
        adjusted_sorted
    )

    adjusted_valid[order] = adjusted_sorted

    adjusted[valid_indices] = adjusted_valid

    return adjusted


# =====================================================
# 6. FEATURE ASSOCIATIONS
# =====================================================

def feature_associations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate associations between the numeric predictors
    and the two outcome variables.

    Cognitive load:
        Spearman rank correlation.

    Binary task success:
        Point-biserial correlation.
    """

    validate_analysis_data(df)

    records = []

    for feature in PREDICTOR_COLUMNS:

        for target in OUTCOME_COLUMNS:

            pair = df[
                [feature, target]
            ].copy()

            pair[feature] = pd.to_numeric(
                pair[feature],
                errors="coerce",
            )

            pair[target] = pd.to_numeric(
                pair[target],
                errors="coerce",
            )

            pair = pair.dropna()

            n = len(pair)

            record = {
                "feature": feature,
                "target": target,
                "method": (
                    "Spearman"
                    if target == "cognitive_load"
                    else "Point-biserial"
                ),
                "n": n,
                "correlation": np.nan,
                "p_value": np.nan,
            }

            if n >= 3 and pair[feature].nunique() > 1:

                if target == "cognitive_load":

                    if pair[target].nunique() > 1:
                        test = spearmanr(
                            pair[feature],
                            pair[target],
                        )

                        record["correlation"] = float(
                            test.statistic
                        )

                        record["p_value"] = float(
                            test.pvalue
                        )

                else:

                    if pair[target].nunique() == 2:

                        test = pointbiserialr(
                            pair[target].astype(int),
                            pair[feature],
                        )

                        record["correlation"] = float(
                            test.statistic
                        )

                        record["p_value"] = float(
                            test.pvalue
                        )

            records.append(record)

    results = pd.DataFrame(records)

    # Adjust the seven tests for each target separately.
    results["adjusted_p_value"] = np.nan

    for target in OUTCOME_COLUMNS:

        mask = results["target"] == target

        results.loc[
            mask,
            "adjusted_p_value",
        ] = benjamini_hochberg(
            results.loc[
                mask,
                "p_value",
            ].to_numpy()
        )

    results["significant_at_0_05_fdr"] = (
        results["adjusted_p_value"] < 0.05
    )

    return results.sort_values(
        ["target", "adjusted_p_value"],
        na_position="last",
    ).reset_index(drop=True)


# =====================================================
# 7. COMPLETE ANALYSIS WORKFLOW
# =====================================================

def run_analysis(
    input_path=FEATURED_DATA_PATH,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Load the featured dataset and save statistical results.

    Returns:
        summary, group_test, associations
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Featured dataset not found: {input_path}\n"
            "Run python -m src.cleaning and "
            "python -m src.features first."
        )

    df = pd.read_csv(input_path)

    validate_analysis_data(df)

    summary = summarize_by_success(df)

    group_test = compare_cognitive_load(df)

    associations = feature_associations(df)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        REPORT_DIR / "cognitive_load_by_success.csv",
        index=False,
    )

    pd.DataFrame(
        [group_test]
    ).to_csv(
        REPORT_DIR / "cognitive_load_group_test.csv",
        index=False,
    )

    associations.to_csv(
        REPORT_DIR / "feature_associations.csv",
        index=False,
    )

    print("=" * 60)
    print("COGNISTREAM — STATISTICAL ANALYSIS")
    print("=" * 60)

    print("\nCognitive load by outcome:")
    print(summary.to_string(index=False))

    print("\nMann–Whitney U test:")
    for key, value in group_test.items():
        print(f"{key}: {value}")

    print("\nFeature associations:")
    print(associations.to_string(index=False))

    print("\nReports saved to:", REPORT_DIR)

    return summary, group_test, associations


if __name__ == "__main__":
    run_analysis()
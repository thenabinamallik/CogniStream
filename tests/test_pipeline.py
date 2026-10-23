"""
CogniStream — Automated Pipeline Tests

Run from the project root:
    python -m pytest tests/ -v

Tests use controlled sample data and do not modify the real CSV.
"""

import numpy as np
import pandas as pd
import pytest

from src.schema import EXPECTED_COLUMNS
from src.data_collection import validate_schema
from src.cleaning import clean_dataset
from src.features import engineer_features
from src.analysis import (
    validate_analysis_data,
    summarize_by_success,
    compare_cognitive_load,
    feature_associations,
)


# =====================================================
# 1. CONTROLLED TEST DATA
# =====================================================

@pytest.fixture
def sample_data():
    """Return a small, valid dataset for testing."""

    return pd.DataFrame({
        "hours_coding": [4.0, 6.0, 8.0, 5.0, 7.0, 3.0],
        "coffee_intake_mg": [100, 150, 200, 80, 180, 50],
        "distractions": [1, 3, 5, 2, 4, 1],
        "sleep_hours": [8.0, 7.0, 6.0, 8.5, 7.5, 9.0],
        "commits": [5, 8, 10, 4, 9, 3],
        "bugs_reported": [1, 2, 3, 1, 2, 0],
        "ai_usage_hours": [0.5, 1.0, 2.0, 0.2, 1.5, 0.0],
        "cognitive_load": [3, 6, 8, 4, 7, 2],
        "task_success": [1, 1, 0, 1, 0, 0],
    })


# =====================================================
# 2. SCHEMA TESTS
# =====================================================

def test_valid_schema(sample_data):
    result = validate_schema(sample_data)

    assert result["schema_valid"] is True
    assert result["missing_columns"] == []


def test_missing_column_is_detected(sample_data):
    incomplete = sample_data.drop(
        columns=["sleep_hours"]
    )

    result = validate_schema(incomplete)

    assert result["schema_valid"] is False
    assert "sleep_hours" in result["missing_columns"]


# =====================================================
# 3. CLEANING TESTS
# =====================================================

def test_cleaning_preserves_valid_rows(sample_data):
    cleaned = clean_dataset(sample_data)

    assert len(cleaned) == len(sample_data)
    assert list(cleaned.columns) == EXPECTED_COLUMNS


def test_cleaning_does_not_modify_input(sample_data):
    original = sample_data.copy(deep=True)

    clean_dataset(sample_data)

    pd.testing.assert_frame_equal(
        sample_data,
        original,
    )


def test_invalid_cognitive_load_becomes_missing(sample_data):
    sample = sample_data.copy()
    sample.loc[0, "cognitive_load"] = 15

    cleaned = clean_dataset(sample)

    assert pd.isna(
        cleaned.loc[0, "cognitive_load"]
    )


def test_invalid_binary_target_becomes_missing(sample_data):
    sample = sample_data.copy()
    sample.loc[0, "task_success"] = 3

    cleaned = clean_dataset(sample)

    assert pd.isna(
        cleaned.loc[0, "task_success"]
    )


# =====================================================
# 4. FEATURE ENGINEERING TESTS
# =====================================================

def test_feature_engineering_preserves_row_count(sample_data):
    featured = engineer_features(sample_data)

    assert len(featured) == len(sample_data)


def test_feature_engineering_adds_expected_features(sample_data):
    featured = engineer_features(sample_data)

    expected = [
        "coding_hours_band",
        "sleep_band",
        "cognitive_load_band",
        "distraction_band",
        "distractions_per_coding_hour",
        "commits_per_coding_hour",
        "bugs_per_commit",
        "ai_hours_per_coding_hour",
        "task_success_label",
    ]

    for column in expected:
        assert column in featured.columns


def test_zero_coding_hours_produces_missing_rates(sample_data):
    sample = sample_data.copy()
    sample.loc[0, "hours_coding"] = 0

    featured = engineer_features(sample)

    assert pd.isna(
        featured.loc[0, "distractions_per_coding_hour"]
    )

    assert pd.isna(
        featured.loc[0, "commits_per_coding_hour"]
    )


# =====================================================
# 5. STATISTICAL ANALYSIS TESTS
# =====================================================

def test_analysis_data_validation(sample_data):
    validate_analysis_data(sample_data)


def test_invalid_task_success_is_rejected(sample_data):
    sample = sample_data.copy()
    sample.loc[0, "task_success"] = 5

    with pytest.raises(ValueError):
        validate_analysis_data(sample)


def test_group_summary_has_both_outcomes(sample_data):
    summary = summarize_by_success(sample_data)

    assert set(summary["task_outcome"]) == {
        "Successful",
        "Unsuccessful",
    }


def test_group_comparison_returns_expected_keys(sample_data):
    result = compare_cognitive_load(sample_data)

    assert "p_value" in result
    assert "u_statistic" in result
    assert "rank_biserial_effect_size" in result

    assert 0 <= result["p_value"] <= 1


def test_associations_include_both_targets(sample_data):
    results = feature_associations(sample_data)

    assert set(results["target"]) == {
        "cognitive_load",
        "task_success",
    }

    assert len(results) == 14
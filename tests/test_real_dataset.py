from pathlib import Path

import pandas as pd

from src.schema import EXPECTED_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


def test_raw_dataset_exists():
    path = ROOT / "dataset" / "raw" / "ai_dev_productivity.csv"

    assert path.exists(), f"Missing dataset: {path}"


def test_raw_dataset_schema_and_integrity():
    path = ROOT / "dataset" / "raw" / "ai_dev_productivity.csv"
    df = pd.read_csv(path)

    assert len(df) == 500
    assert list(df.columns) == EXPECTED_COLUMNS
    assert not df.isna().any().any()
    assert not df.duplicated().any()


def test_processed_dataset_preserves_record_count():
    path = ROOT / "dataset" / "processed" / "cleaned_data.csv"

    assert path.exists(), (
        "Run python -m src.cleaning before this test."
    )

    df = pd.read_csv(path)

    assert len(df) == 500
    assert list(df.columns) == EXPECTED_COLUMNS


def test_featured_dataset_contains_engineered_columns():
    path = ROOT / "dataset" / "processed" / "featured_data.csv"

    assert path.exists(), (
        "Run python -m src.features before this test."
    )

    df = pd.read_csv(path)

    expected_features = [
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

    assert len(df) == 500

    for column in expected_features:
        assert column in df.columns
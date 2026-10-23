# CogniStream

**Developer Flow-State & Cognitive Load Analytics**

CogniStream is a data analytics project that explores how developer work patterns relate to reported cognitive load and task success. It includes data validation, cleaning, feature engineering, exploratory data analysis (EDA), statistical association testing, automated tests, and a Streamlit dashboard.

> **Interpretation note:** This project analyzes a dataset of 500 records. The available data does not contain timestamps, developer identifiers, or session boundaries. Therefore, the project can explore patterns associated with cognitive load and task success, but it cannot directly measure flow-state duration, change over time, or causal effects.

## Project goals

- Validate the raw dataset against an expected schema and basic value ranges.
- Keep the raw input unchanged and produce a cleaned dataset for analysis.
- Create analyst-defined bands and activity-rate features.
- Explore distributions, group differences, and relationships between work-pattern variables, cognitive load, and task success.
- Correct families of statistical p-values for multiple comparisons using the Benjamini–Hochberg procedure.
- Present interactive summaries in a Streamlit dashboard.
- Verify core pipeline behavior with automated tests.

## Dataset

The raw file is expected at `dataset/raw/ai_dev_productivity.csv` and contains these nine columns:

| Column | Description |
|---|---|
| `hours_coding` | Hours spent coding for a record |
| `coffee_intake_mg` | Reported caffeine intake in milligrams |
| `distractions` | Number of distractions recorded |
| `sleep_hours` | Reported hours of sleep |
| `commits` | Number of commits recorded |
| `bugs_reported` | Number of bugs reported |
| `ai_usage_hours` | Hours of AI tool usage |
| `cognitive_load` | Cognitive-load score on a 1–10 scale |
| `task_success` | Binary task outcome: `0` or `1` |

The project profile previously reported 500 rows, no missing values, no duplicate rows, and no schema or numeric-range violations. Re-run the data-collection command to inspect the current dataset; the notebook `notebooks/01_data_collection.ipynb` also contains the profile-export step.

## Repository structure

```text
CogniStream/
├── dataset/
│   ├── raw/
│   │   └── ai_dev_productivity.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── featured_data.csv
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_analysis.ipynb
├── reports/
│   ├── data_collection_profile.json
│   ├── cleaning_report.csv
│   ├── cognitive_load_by_success.csv
│   ├── cognitive_load_group_test.csv
│   ├── feature_associations.csv
│   ├── methodology.md
│   └── final_findings_template.md
├── src/
│   ├── schema.py
│   ├── data_collection.py
│   ├── cleaning.py
│   ├── features.py
│   └── analysis.py
├── dashboard/
│   └── app.py
├── tests/
│   ├── test_pipeline.py
│   └── test_real_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

Some report files are generated when pipeline commands run, so they might not exist in a fresh clone until the pipeline is executed.

## Setup

Use Python 3.11 or a compatible Python version. Run commands from the **project root** (`CogniStream`), not from inside `src`, `tests`, or `notebooks`.

### 1. Create and activate an environment (recommended)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, activate the environment using your preferred Python environment manager or call `.venv\Scripts\python.exe` directly.

### 2. Check the raw dataset and schema

```powershell
python -m src.data_collection
```

### 3. Run cleaning and feature engineering

```powershell
python -m src.cleaning
python -m src.features
```

Expected generated datasets:

- `dataset/processed/cleaned_data.csv`
- `dataset/processed/featured_data.csv`

The raw CSV is not overwritten by the cleaning step. Invalid numeric or out-of-range values are handled in the processed output according to the project rules.

### 4. Run statistical analysis

```powershell
python -m src.analysis
```

Expected analysis outputs:

- `reports/cognitive_load_by_success.csv`
- `reports/cognitive_load_group_test.csv`
- `reports/feature_associations.csv`

### 5. Run the automated tests

```powershell
python -m pytest tests/ -v
```

Run pytest from the project root. This avoids import-path errors such as `ModuleNotFoundError: No module named 'src'` that can occur when a test file is launched directly from the `tests` directory.

### 6. Launch the dashboard

```powershell
streamlit run dashboard/app.py
```

If the dashboard reports that processed data is missing, run the cleaning and feature-engineering steps above first.

### 7. Open the notebooks

```powershell
jupyter lab
```

Run notebooks in order: data collection, cleaning, EDA, then statistical analysis. The notebooks use Matplotlib for static analysis plots, while the Streamlit dashboard uses interactive Plotly charts.

## Engineered features

The feature pipeline adds the following fields to the source columns:

- `coding_hours_band`
- `sleep_band`
- `cognitive_load_band`
- `distraction_band`
- `distractions_per_coding_hour`
- `commits_per_coding_hour`
- `bugs_per_commit`
- `ai_hours_per_coding_hour`
- `task_success_label`

Bands are analyst-defined descriptive categories, not clinically validated or universal thresholds. Rate features may be undefined when the denominator is zero; such values should be treated as undefined rather than interpreted as zero performance.

## Statistical approach

The analysis includes descriptive summaries, a Mann–Whitney U test comparing cognitive-load distributions between task-outcome groups, rank-based effect-size reporting, Spearman rank correlations for numeric predictors against cognitive load, point-biserial correlations for numeric predictors against binary task success, and Benjamini–Hochberg adjusted p-values within target-specific comparison families.

Read [`reports/methodology.md`](reports/methodology.md) before interpreting the results. Statistical association is not proof of causation, and results depend on how this dataset was collected and represented.

## Main outputs

| Output | Purpose |
|---|---|
| `reports/data_collection_profile.json` | Dataset profile exported by the data-collection notebook |
| `reports/cleaning_report.csv` | Summary of values handled by the cleaning pipeline |
| `reports/cognitive_load_by_success.csv` | Cognitive-load summary grouped by task outcome |
| `reports/cognitive_load_group_test.csv` | Test results for comparing cognitive load between task-outcome groups |
| `reports/feature_associations.csv` | Feature association estimates and p-value adjustments |
| `dashboard/app.py` | Interactive, filterable dashboard |

Use [`reports/final_findings_template.md`](reports/final_findings_template.md) to write the final interpretation after generating and reviewing the report files. Do not fill in results from assumptions or from the example/test fixture.

## Limitations

1. The data has no timestamps or session IDs, so temporal patterns and flow-state duration cannot be evaluated directly.
2. The meaning and collection method for each measurement are not documented in the CSV schema alone; confirm these with the dataset provider before making strong claims.
3. Observational patterns do not establish that caffeine, sleep, AI usage, distractions, or any other feature caused an outcome.
4. Sample size, sampling bias, data-generation process, and measurement error may limit generalizability.
5. Analyst-defined feature bands are intended for exploration, not diagnosis or professional assessment.
6. The dataset's `task_success` field is interpreted as binary, with values `0` and `1`; confirm the original label definitions before publishing results.

## Reproducibility checklist

- [ ] Keep the original raw CSV unchanged.
- [ ] Run data collection, cleaning, feature engineering, and analysis from the project root.
- [ ] Run `python -m pytest tests/ -v` and retain the output.
- [ ] Generate the report CSV files from the current dataset.
- [ ] Record the date, Python version, and dependency versions used for final reporting.
- [ ] Base findings only on the current generated report values.
- [ ] Distinguish descriptive patterns, statistical associations, and causal claims.

## Project status

This is an exploratory data analytics project. It should be described as cognitive-load and task-outcome analysis unless additional timestamped, session-level data is added to support direct flow-state measurement.

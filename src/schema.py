"""
CogniStream — Dataset Schema

Defines the official columns, feature categories,
and validation rules for the AI developer productivity dataset.
"""

PROJECT_NAME = "CogniStream"

DATASET_NAME = "AI Developer Productivity Dataset"


# --------------------------------------------------
# 1. OFFICIAL DATASET COLUMNS
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "hours_coding",
    "coffee_intake_mg",
    "distractions",
    "sleep_hours",
    "commits",
    "bugs_reported",
    "ai_usage_hours",
    "cognitive_load",
    "task_success",
]


# --------------------------------------------------
# 2. FEATURE CLASSIFICATION
# --------------------------------------------------

CONTINUOUS_FEATURES = [
    "hours_coding",
    "coffee_intake_mg",
    "sleep_hours",
    "ai_usage_hours",
]

COUNT_FEATURES = [
    "distractions",
    "commits",
    "bugs_reported",
]

COGNITIVE_FEATURE = "cognitive_load"

TARGET_FEATURE = "task_success"


# --------------------------------------------------
# 3. EXPECTED VALUE CONSTRAINTS
# --------------------------------------------------

NON_NEGATIVE_COLUMNS = [
    "hours_coding",
    "coffee_intake_mg",
    "distractions",
    "sleep_hours",
    "commits",
    "bugs_reported",
    "ai_usage_hours",
]

# These are general validity rules, not observed
# dataset minima and maxima.
MAX_DAILY_HOURS = 24

COGNITIVE_LOAD_MIN = 1
COGNITIVE_LOAD_MAX = 10

TASK_SUCCESS_VALUES = [0, 1]


# --------------------------------------------------
# 4. DATASET METADATA
# --------------------------------------------------

COGNITIVE_LOAD_DESCRIPTION = (
    "Reported cognitive load on a scale from 1 to 10."
)

TASK_SUCCESS_DESCRIPTION = (
    "Binary target: 1 indicates success; "
    "0 indicates failure."
)

DATA_SOURCE_TYPE = "user_provided_csv"
"""
CogniStream: 

Interactive Streamlit dashboard using the real
AI Developer Productivity dataset.

Features:
- Interactive sidebar filters
- Dataset overview and KPIs
- Cognitive-load distribution
- Task-success comparisons
- Developer activity analysis
- Correlation heatmap
- Statistical association results
- Filtered data exploration and CSV export

Limitations:
- Observations are not necessarily unique developers or sessions.
- Activity indicators do not directly measure psychological flow.
- Associations do not establish causation.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.schema import EXPECTED_COLUMNS
from src.features import (
    engineer_features,
    CLEANED_DATA_PATH,
    FEATURED_DATA_PATH,
)
from src.analysis import (
    validate_analysis_data,
    summarize_by_success,
    compare_cognitive_load,
    feature_associations,
)


# =====================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CogniStream | Developer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 3. LOAD DATA
# =====================================================

@st.cache_data
def load_dashboard_data():
    """
    Load the featured dataset when available.
    Otherwise, derive features from the cleaned dataset.
    """

    if FEATURED_DATA_PATH.exists():
        df = pd.read_csv(FEATURED_DATA_PATH)

    elif CLEANED_DATA_PATH.exists():
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
        df = engineer_features(cleaned_df)

    else:
        raise FileNotFoundError(
            "Processed data was not found.\n\n"
            "Run these commands from your project root:\n"
            "python -m src.cleaning\n"
            "python -m src.features"
        )

    # Check the official input features.
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Normalize numeric columns for downstream analysis.
    for column in EXPECTED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    validate_analysis_data(df)

    return df


try:
    data = load_dashboard_data()

except Exception as error:
    st.title("CogniStream")
    st.error("Unable to load the dashboard dataset.")
    st.code(str(error))
    st.stop()


# =====================================================
# 4. DASHBOARD HEADER
# =====================================================

st.title("CogniStream")
st.subheader("Developer Flow-State & Cognitive Load Analytics")

st.markdown(
    """
    Explore developer activity, self-reported cognitive load,
    and task outcomes using an interactive data analytics dashboard.
    """
)

st.info(
    "Dataset notice: This dashboard analyzes the supplied "
    "AI Developer Productivity dataset. Interpretations depend "
    "on its sampling and measurement methodology. Activity-based "
    "metrics are not direct measurements of psychological flow."
)


# =====================================================
# 5. SIDEBAR FILTERS
# =====================================================

st.sidebar.title("Dashboard Filters")

filtered = data.copy()

# -----------------------------------------------------
# Task outcome filter
# -----------------------------------------------------

available_outcomes = {
    0: "Unsuccessful",
    1: "Successful",
}

selected_outcomes = st.sidebar.multiselect(
    "Task outcome",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda value: available_outcomes[value],
)

filtered = filtered[
    filtered["task_success"].isin(selected_outcomes)
]


# -----------------------------------------------------
# Numeric range filters
# -----------------------------------------------------

def apply_range_filter(
    df,
    column,
    label,
    enabled=True,
):
    """
    Apply a sidebar range filter to a numeric column.
    """

    if not enabled:
        return df

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return df

    minimum = float(values.min())
    maximum = float(values.max())

    if minimum == maximum:
        st.sidebar.caption(
            f"{label}: all available values equal {minimum:g}"
        )
        return df

    is_integer = np.allclose(
        values.to_numpy(),
        np.round(values.to_numpy()),
    )

    step = (
        1.0
        if is_integer
        else max((maximum - minimum) / 100, 0.01)
    )

    selected_range = st.sidebar.slider(
        label,
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum),
        step=step,
    )

    return df[
        pd.to_numeric(
            df[column],
            errors="coerce",
        ).between(
            selected_range[0],
            selected_range[1],
        )
    ]


filtered = apply_range_filter(
    filtered,
    "cognitive_load",
    "Cognitive load",
)

filtered = apply_range_filter(
    filtered,
    "hours_coding",
    "Coding hours",
)

filtered = apply_range_filter(
    filtered,
    "sleep_hours",
    "Sleep hours",
)

filtered = apply_range_filter(
    filtered,
    "distractions",
    "Distractions",
)

filtered = apply_range_filter(
    filtered,
    "ai_usage_hours",
    "AI usage hours",
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Changing filters recalculates the displayed metrics "
    "and comparisons for the selected observations."
)


# =====================================================
# 6. KPI CARDS
# =====================================================

st.markdown("## Overview")

if filtered.empty:

    st.warning(
        "No observations match the selected filters. "
        "Adjust the sidebar selections."
    )

    st.stop()


total_observations = len(filtered)

success_rate = (
    filtered["task_success"].mean() * 100
)

mean_cognitive_load = (
    filtered["cognitive_load"].mean()
)

median_coding_hours = (
    filtered["hours_coding"].median()
)

mean_ai_usage = (
    filtered["ai_usage_hours"].mean()
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Observations",
    f"{total_observations:,}",
)

k2.metric(
    "Task success rate",
    f"{success_rate:.1f}%",
)

k3.metric(
    "Mean cognitive load",
    f"{mean_cognitive_load:.2f}/10",
)

k4.metric(
    "Median coding hours",
    f"{median_coding_hours:.2f} h",
)

st.caption(
    "Metrics are calculated from the currently filtered observations. "
    "Task success rate describes the dataset, not an individual's "
    "personal success probability."
)


# =====================================================
# 7. SUMMARY TABS
# =====================================================

tab_overview, tab_activity, tab_statistics, tab_data = st.tabs(
    [
        "Overview",
        "Activity Analysis",
        "Statistical Insights",
        "Data Explorer",
    ]
)


# =====================================================
# 8. OVERVIEW TAB
# =====================================================

with tab_overview:

    st.markdown("### Cognitive-load distribution")

    fig_load = px.histogram(
        filtered,
        x="cognitive_load",
        nbins=10,
        title="Reported Cognitive Load",
        labels={
            "cognitive_load": "Cognitive load (1–10)",
            "count": "Observations",
        },
    )

    fig_load.update_layout(
        template="plotly_white",
    )

    st.plotly_chart(
        fig_load,
        use_container_width=True,
    )

    st.markdown("### Task-success distribution")

    success_summary = (
        filtered["task_success"]
        .map(available_outcomes)
        .value_counts()
        .rename_axis("Task outcome")
        .reset_index(name="Observations")
    )

    fig_success = px.bar(
        success_summary,
        x="Task outcome",
        y="Observations",
        color="Task outcome",
        text="Observations",
        title="Successful vs Unsuccessful Outcomes",
    )

    fig_success.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    st.plotly_chart(
        fig_success,
        use_container_width=True,
    )

    st.markdown("### Cognitive load by task outcome")

    fig_box = px.box(
        filtered.assign(
            task_outcome=filtered["task_success"].map(
                available_outcomes
            )
        ),
        x="task_outcome",
        y="cognitive_load",
        color="task_outcome",
        points="outliers",
        title="Cognitive Load vs Task Outcome",
        labels={
            "task_outcome": "Task outcome",
            "cognitive_load": "Reported cognitive load",
        },
    )

    fig_box.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True,
    )


# =====================================================
# 9. ACTIVITY ANALYSIS TAB
# =====================================================

with tab_activity:

    st.markdown("### Coding and cognitive load")

    fig_coding = px.scatter(
        filtered,
        x="hours_coding",
        y="cognitive_load",
        color="task_success",
        hover_data=[
            "distractions",
            "sleep_hours",
            "coffee_intake_mg",
            "ai_usage_hours",
        ],
        title="Coding Hours vs Cognitive Load",
        labels={
            "hours_coding": "Coding hours",
            "cognitive_load": "Reported cognitive load",
            "task_success": "Task success",
        },
    )

    fig_coding.update_layout(
        template="plotly_white",
    )

    st.plotly_chart(
        fig_coding,
        use_container_width=True,
    )

    st.markdown("### Distractions and cognitive load")

    fig_distractions = px.box(
        filtered,
        x="distractions",
        y="cognitive_load",
        title="Cognitive Load by Distraction Count",
        labels={
            "distractions": "Distractions",
            "cognitive_load": "Reported cognitive load",
        },
    )

    fig_distractions.update_layout(
        template="plotly_white",
    )

    st.plotly_chart(
        fig_distractions,
        use_container_width=True,
    )

    st.markdown("### AI usage and task outcome")

    fig_ai = px.box(
        filtered.assign(
            task_outcome=filtered["task_success"].map(
                available_outcomes
            )
        ),
        x="task_outcome",
        y="ai_usage_hours",
        color="task_outcome",
        title="Recorded AI Usage by Task Outcome",
        labels={
            "task_outcome": "Task outcome",
            "ai_usage_hours": "AI usage hours",
        },
    )

    fig_ai.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    st.plotly_chart(
        fig_ai,
        use_container_width=True,
    )

    st.caption(
        "These charts describe associations in the supplied data. "
        "They do not establish that coding hours, distractions, "
        "or AI usage cause particular outcomes."
    )


# =====================================================
# 10. STATISTICAL INSIGHTS TAB
# =====================================================

with tab_statistics:

    st.markdown("### Cognitive load by task outcome")

    group_summary = summarize_by_success(
        filtered,
    )

    st.dataframe(
        group_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Group comparison")

    test_result = compare_cognitive_load(
        filtered,
    )

    r1, r2, r3 = st.columns(3)

    p_value = test_result["p_value"]
    effect_size = test_result["rank_biserial_effect_size"]
    median_diff = (
        test_result[
            "median_difference_success_minus_unsuccessful"
        ]
    )

    r1.metric(
        "Mann–Whitney p-value",
        "N/A" if p_value is None else f"{p_value:.4f}",
    )

    r2.metric(
        "Rank-biserial effect",
        "N/A" if effect_size is None else f"{effect_size:.3f}",
    )

    r3.metric(
        "Median difference",
        "N/A" if median_diff is None else f"{median_diff:.2f}",
    )

    if p_value is None:
        st.warning(
            "Insufficient observations in one or both outcome groups "
            "to complete the group comparison."
        )

    elif p_value < 0.05:
        st.write(
            "The group comparison provides evidence of a difference "
            "in cognitive-load distributions at the 0.05 significance "
            "level. Consider the effect size and group distributions "
            "before drawing conclusions."
        )

    else:
        st.write(
            "The group comparison does not provide sufficient "
            "evidence of a distributional difference at the 0.05 "
            "significance level. This does not establish equivalence."
        )

    st.markdown("### Feature associations")

    associations = feature_associations(
        filtered,
    )

    st.dataframe(
        associations.round(4),
        use_container_width=True,
        hide_index=True,
    )

    association_plot = associations.dropna(
        subset=["correlation"]
    ).copy()

    if not association_plot.empty:

        association_plot["association_label"] = (
            association_plot["feature"]
            + " → "
            + association_plot["target"]
        )

        association_plot = association_plot.sort_values(
            "correlation"
        )

        fig_association = px.bar(
            association_plot,
            x="correlation",
            y="association_label",
            color="target",
            orientation="h",
            title="Feature Associations",
            labels={
                "correlation": "Correlation coefficient",
                "association_label": "Feature and target",
            },
        )

        fig_association.update_layout(
            template="plotly_white",
        )

        st.plotly_chart(
            fig_association,
            use_container_width=True,
        )

    st.caption(
        "Adjusted p-values use the Benjamini–Hochberg procedure "
        "within each target's family of tests. Statistical significance "
        "does not demonstrate causation."
    )


# =====================================================
# 11. DATA EXPLORER TAB
# =====================================================

with tab_data:

    st.markdown("### Filtered dataset")

    st.write(
        f"Showing {len(filtered):,} of {len(data):,} observations."
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download filtered dataset (CSV)",
        data=csv_data,
        file_name="cognistream_filtered_data.csv",
        mime="text/csv",
    )

    st.markdown("### Dataset information")

    st.write(
        {
            "Original_dataset_rows": len(data),
            "Filtered_rows": len(filtered),
            "Original_columns": len(EXPECTED_COLUMNS),
            "Featured_dataset_columns": len(data.columns),
            "Missing_values_in_filtered_data": int(
                filtered.isna().sum().sum()
            ),
        }
    )


# =====================================================
# 12. CORRELATION ANALYSIS
# =====================================================

st.markdown("---")
st.markdown("## Correlation overview")

correlation_columns = [
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

correlation_matrix = filtered[
    correlation_columns
].corr(method="pearson")

fig_corr = px.imshow(
    correlation_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Pearson Correlation Matrix",
)

fig_corr.update_layout(
    template="plotly_white",
    height=650,
)

st.plotly_chart(
    fig_corr,
    use_container_width=True,
)


# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "CogniStream | Developer Flow-State & Cognitive Load Analytics"
)

st.caption(
    "Descriptive and exploratory analytics only. "
    "The dataset does not directly measure psychological flow, "
    "and the dashboard is not a diagnostic or employee-ranking tool."
)

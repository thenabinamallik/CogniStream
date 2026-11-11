# CogniStream — Final Findings Report

> **Before publishing:** Replace every bracketed placeholder using the outputs generated from the current dataset. Do not infer or invent statistics. If a value is unavailable, say so and explain why.

## 1. Executive summary

This analysis examined **[N] records** to explore relationships between developer activity variables, cognitive-load scores, and task outcomes. The primary observations were: **[Finding 1 based on an actual report]**, **[Finding 2 based on an actual report]**, and **[Finding 3 based on an actual report]**.

These findings are **[descriptive / statistically supported associations / mixed]** and should not be interpreted as evidence of causation.

## 2. Dataset quality

Source file: `dataset/raw/ai_dev_productivity.csv`  
Rows and columns: **[row count] rows × [column count] columns**  
Missing values: **[result from data profile]**  
Duplicate rows: **[result from data profile]**  
Schema and numeric validation: **[result from data profile]**

Data quality notes: **[Any cleaning changes or limitations shown in the generated reports]**.

## 3. Cognitive load and task outcome

Use `reports/cognitive_load_by_success.csv` and `reports/cognitive_load_group_test.csv`.

| Statistic | Task outcome 0 | Task outcome 1 |
|---|---:|---:|
| Number of records | [n0] | [n1] |
| Mean cognitive load | [mean0] | [mean1] |
| Median cognitive load | [median0] | [median1] |
| Standard deviation | [sd0] | [sd1] |

Mann–Whitney U statistic: **[U statistic]**  
P-value: **[p-value]**  
Rank-biserial effect size: **[effect estimate]**

**Interpretation:** [Explain which group's cognitive-load values tend to be higher/lower, taking account of the group summaries, effect-size direction, uncertainty, and exact group ordering used by the code. Avoid claiming that cognitive load caused task success or failure.]

## 4. Feature associations

Use `reports/feature_associations.csv`. State clearly whether each result refers to cognitive load or task success and whether it uses raw or adjusted p-values.

| Target | Feature | Estimate | P-value | Adjusted p-value | Interpretation |
|---|---|---:|---:|---:|---|
| [cognitive_load/task_success] | [feature] | [coefficient] | [p] | [adjusted p] | [Direction and practical interpretation] |
| [cognitive_load/task_success] | [feature] | [coefficient] | [p] | [adjusted p] | [Direction and practical interpretation] |
| [cognitive_load/task_success] | [feature] | [coefficient] | [p] | [adjusted p] | [Direction and practical interpretation] |

Select only the most relevant results and disclose the multiple-comparison adjustment. Correlation coefficients describe association, not causation.

## 5. Exploratory observations

1. **[Observed distribution or group pattern]** — Evidence: [notebook chart/table and relevant statistic].
2. **[Observed distribution or group pattern]** — Evidence: [notebook chart/table and relevant statistic].
3. **[Observed distribution or group pattern]** — Evidence: [notebook chart/table and relevant statistic].

Do not call these findings predictive unless a predictive model has been trained, validated on held-out data, and evaluated with appropriate metrics. The current analysis pipeline is primarily descriptive and inferential.

## 6. Recommendations

Recommendations should follow directly from the reported evidence. Possible categories to consider, only if supported by results, include improving how work sessions and task outcomes are recorded, exploring a result in a larger representative sample, or collecting more contextual data before drawing conclusions.

- **Evidence-backed recommendation:** [Recommendation] — supported by [result].
- **Further validation:** [What should be tested with more data or a study design].
- **Data collection improvement:** [Specific missing field or definition to record].

## 7. Limitations

- The dataset does not contain timestamps, session IDs, or developer IDs, so temporal trends and flow-state duration cannot be directly measured.
- Cognitive load is a score in the dataset; it is not automatically equivalent to a validated flow-state measure.
- Observational associations cannot demonstrate causality.
- Results may be affected by sampling, measurement, missing context, and the unknown data-collection process.
- Analyst-defined feature bands are exploratory categories, not universal thresholds.

## 8. Conclusion

Based on the generated results, **[summarize the strongest defensible conclusion in one or two sentences]**. The analysis supports **[a descriptive observation / an association requiring further validation]**, while it does not establish **[unsupported causal or flow-state claim]**. The next step is **[a specific, evidence-based follow-up]**.

## 9. Reproducibility record

- Analysis date: [YYYY-MM-DD]
- Dataset file/version: [filename and version/hash if available]
- Python version: [version]
- Dependency environment: [requirements file or environment details]
- Test command and result: `python -m pytest tests/ -v` — [pass/fail and summary]
- Reports regenerated: [list files]

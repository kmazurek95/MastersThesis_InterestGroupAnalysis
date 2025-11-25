# Analysis: Reproducing Thesis Results

**Purpose**: Reproducible analysis scripts and notebooks starting from the `multi_level_data` dataset

---

## Overview

This directory contains all code needed to reproduce the thesis findings using the validated `multi_level_data` dataset. All analyses start from this dataset—there is no need to collect or process raw data.

---

## Structure

```
analysis/
├── README.md                          # This file
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_descriptive_statistics.ipynb
│   ├── 03_reproduce_thesis_results.ipynb
│   └── 04_extended_analyses.ipynb
└── scripts/                           # Standalone scripts
    ├── multilevel_models.R
    └── visualizations.py
```

---

## Quick Start

### Prerequisites

```bash
# Python environment
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter pyarrow

# R environment (for GLMM models)
# In R console:
install.packages(c("lme4", "ggplot2", "broom.mixed", "arrow", "dplyr"))
```

### Run Analysis Notebooks

```bash
# Start Jupyter
jupyter notebook analysis/notebooks/

# Run notebooks in order:
# 1. 01_data_exploration.ipynb - Load and explore the dataset
# 2. 02_descriptive_statistics.ipynb - Summary tables and basic visualizations
# 3. 03_reproduce_thesis_results.ipynb - Replicate all thesis findings
# 4. 04_extended_analyses.ipynb - Additional analyses and robustness checks
```

### Run Statistical Models

```bash
# Run R multilevel models
Rscript analysis/scripts/multilevel_models.R

# Generate Python visualizations
python analysis/scripts/visualizations.py
```

---

## Notebooks

### 01_data_exploration.ipynb

- Load the `multi_level_data` dataset
- Explore data structure and variables
- Check data quality (missing values, outliers)
- Generate summary statistics

### 02_descriptive_statistics.ipynb

- Descriptive tables for variables
- Prominence rates by policy area, group type, year
- Distributions of key predictors
- Correlation matrices

### 03_reproduce_thesis_results.ipynb

- Reproduce all thesis tables and figures
- Multilevel model results
- Model diagnostics and fit statistics
- Compare with published results

### 04_extended_analyses.ipynb

- Alternative model specifications
- Robustness checks
- Additional hypotheses
- Exploratory visualizations

---

## Scripts

### multilevel_models.R

Generalized Linear Mixed-Effects Models (GLMM) for prominence:

```r
# Model specification from thesis
model <- glmer(
    prominent ~
        lobbying_expenditure + policy_breadth + membership_size +
        speaker_seniority + speaker_party +
        policy_salience +
        (1 | policy_area) + (1 | interest_group_name),
    data = data,
    family = binomial
)
```

**Outputs**:
- Model summaries
- Coefficient tables
- Variance components
- Model diagnostics

### visualizations.py

Publication-ready figures:

- Prominence by policy area (bar chart)
- Lobbying expenditure vs. prominence (scatter)
- Temporal trends (line plots)
- Model coefficient plots

**Outputs** saved to `outputs/figures/`

---

## Expected Outputs

Running all analyses should generate:

### Figures
- `fig1_prominence_by_policy_area.png`
- `fig2_lobbying_expenditure_scatter.png`
- `fig3_temporal_trends.png`
- `fig4_model_coefficients.png`

### Tables
- `table1_descriptive_statistics.csv`
- `table2_glmm_results.tex`
- `table3_robustness_checks.html`

---

## Validation

To validate that you've reproduced the thesis results:

1. **Compare coefficients**: GLMM coefficients should match thesis Table 3 within rounding error
2. **Check significance**: Same variables should be significant at p < 0.05
3. **Verify N**: Sample size should be consistent (check for missing data handling)
4. **Compare figures**: Visual inspection of plots against thesis figures

---

## Extending the Analysis

Want to test new hypotheses? Here's how:

### Adding New Variables

```python
# Load dataset
import pandas as pd
data = pd.read_parquet('../data/multi_level_data/multi_level_data_v1.0.parquet')

# Add new variables
data['new_variable'] = ...

# Re-run models with new variable
```

### Alternative Models

```r
# Try different random effects structure
model_alt <- glmer(
    prominent ~ ... + (1 + lobbying_expenditure | policy_area),
    data = data,
    family = binomial
)

# Compare models
anova(model, model_alt)
```

---

## Reproducibility Checklist

Before publishing results based on these analyses:

- [ ] Loaded data from `multi_level_data_v1.0.parquet`
- [ ] Documented any data transformations or filtering
- [ ] Recorded package versions (`sessionInfo()` in R, `pip freeze` in Python)
- [ ] Saved all outputs with version numbers
- [ ] Compared results to thesis findings
- [ ] Noted any deviations or unexpected results

---

## Troubleshooting

### Dataset not found

See `../data/multi_level_data/README.md` for access instructions.

### Models won't converge

Try:
- Scaling continuous variables
- Simpler random effects structure
- Different optimizers (`bobyqa`, `nloptwrap`)

### Missing packages

```bash
# Python
pip install -r requirements.txt

# R
install.packages(c("lme4", "ggplot2", "broom.mixed", "arrow"))
```

---

## Contact

For questions about reproducing results:

**Kaleb Mazurek**
- Email: kalebmazurek@gmail.com
- GitHub: [@kmazurek95](https://github.com/kmazurek95)

---

**Last Updated**: November 25, 2024

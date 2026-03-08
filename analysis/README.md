# Analysis

Reproducible analysis scripts starting from the `multi_level_data` dataset.

## Notebooks

### 01_Exploratory_Prominence_Analysis.ipynb

Exploratory analysis of prominence patterns:
- Concentration and distribution of prominent mentions
- Top organizations by prominence
- Partisan patterns in group citations
- Issue area analysis

### 02_Statistical_Models.ipynb

Logistic regression models testing three hypotheses:
- Model A: Issue salience and strategic communication
- Model B: Politician characteristics and electoral positioning
- Model C: Organizational resources and lobbying capacity

### 03_Multilevel_Models.Rmd

Generalized Linear Mixed-Effects Models (GLMM) in R with crossed random effects for organization and policy area. This is the primary inferential analysis from the thesis.

## Python Dependencies

```bash
pip install -r ../requirements.txt
```

## R Dependencies

The R Markdown notebook requires:

```r
install.packages(c(
  "tidyverse",    # Data manipulation
  "lme4",         # Mixed-effects models
  "lmerTest",     # p-values for mixed models
  "broom.mixed",  # Tidy model outputs
  "performance",  # Model diagnostics
  "see",          # Visualization for performance
  "car",          # Companion to Applied Regression
  "ggeffects",    # Marginal effects plots
  "sjPlot",       # Model summary tables
  "patchwork",    # Plot composition
  "knitr",        # Report generation
  "kableExtra",   # Table formatting
  "gt",           # Publication tables
  "rmarkdown"     # Rendering
))
```

## Data

All notebooks load data from `../data/multi_level_data/`. The main file is `level1_FINAL.csv` (19,165 mentions with 175 variables).

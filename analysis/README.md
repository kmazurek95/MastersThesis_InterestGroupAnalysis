# Analysis Notebooks

These notebooks reproduce the thesis analysis starting from the validated dataset in `../data/multi_level_data/`.

## Notebooks

- **01_Exploratory_Prominence_Analysis.ipynb**: Descriptive analysis of prominence concentration, partisan patterns, and issue area distribution.
- **02_Statistical_Models.ipynb**: Python implementation of mixed-effects logistic regression (Models A, B, C). Comparison implementation; the primary models are in R.
- **03_Multilevel_Models.Rmd**: Primary inferential analysis using lme4 GLMMs with crossed random effects for organization and policy area. This is what the thesis findings are based on.

## Dependencies

Python: `pip install -r ../requirements.txt`

R: `lme4`, `tidyverse`, `broom.mixed`, `performance`, `ggeffects`, `sjPlot`, `knitr`, `kableExtra`, `gt`

## Data

All notebooks load from `../data/multi_level_data/level1_FINAL.csv` (19,165 mentions, 175 variables). Run `git lfs pull` first.

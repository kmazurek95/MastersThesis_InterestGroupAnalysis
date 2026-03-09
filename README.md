# Interest Group Prominence in Congressional Speech

MSc Thesis, University of Amsterdam, 2024

I studied why some advocacy organizations get recognized as authoritative voices in U.S. congressional floor speeches while others are mentioned only in passing. Using NLP text classification and multilevel regression models, I analyzed prominence patterns across the 114th and 115th Congress.

## Key Findings

- Lobbying expenditure and broad policy engagement strongly predict whether a group is cited as an authority
- Medium-salience policy areas show higher advocacy visibility than high-salience ones
- Politician seniority has an unexpected negative effect on prominence
- The top 5% of organizations account for over 50% of all prominent mentions

## Thesis Documents

| Document | Description |
|----------|-------------|
| [Full Thesis (PDF)](legacy/5.%20Visualization%20and%20Reporting/Thesis_UvA_Kaleb_Mazurek.pdf) | Complete thesis manuscript |

## Data

- ~20,699 mentions of interest groups in congressional floor speeches (pre-filtering total)
- Analytical sample: 19,165 mentions after dropping cases with undetermined policy areas
- 5,323 organizations from the Washington Representatives Study
- 114th and 115th Congress (2015-2019)
- Full-sample design: includes organizations with zero mentions as the baseline

## Methods

- Text classification: Naive Bayes / SVM (F1 = 0.79)
- Statistical models: Generalized Linear Mixed-Effects Models (GLMMs) in R
- Random effects: Organization and Policy Area

## Analysis Notebooks

| Notebook | What it does |
|----------|-------------|
| [01_Exploratory_Prominence_Analysis.ipynb](analysis/01_Exploratory_Prominence_Analysis.ipynb) | Concentration, distribution, partisan patterns |
| [02_Statistical_Models.ipynb](analysis/02_Statistical_Models.ipynb) | Python implementation of GLMM models A, B, C |
| [03_Multilevel_Models.Rmd](analysis/03_Multilevel_Models.Rmd) | Primary R implementation with lme4 (thesis models) |

## Repository Structure

```
analysis/          Jupyter notebooks and R Markdown for analysis
legacy/            Original pipeline code (data collection, processing, classification, modeling)
data/              Analysis datasets (managed via Git LFS)
```

## How to Run

```
git clone https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis.git
cd MastersThesis_InterestGroupAnalysis
pip install -r requirements.txt
jupyter notebook analysis/01_Exploratory_Prominence_Analysis.ipynb
```

For R models: install `lme4`, `ggplot2`, `broom.mixed`, `readr`, `dplyr`, `knitr`, then open `analysis/03_Multilevel_Models.Rmd` in RStudio.

Data files are tracked with Git LFS. Run `git lfs pull` if CSVs appear as pointer files.

## Related Work

A modernized re-implementation of this pipeline with improved extraction and classification (F1 = 0.91, kappa = 0.84) is available at [ThesisPipelineRework](https://github.com/kmazurek95/ThesisPipelineRework).

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata.

## License

MIT License - see [LICENSE](LICENSE).

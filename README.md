# Interest Group Prominence in Congressional Speech

MSc Thesis — University of Amsterdam, 2023

**[Read the full thesis (PDF)](docs/Thesis_UvA_Kaleb_Mazurek.pdf)**

I studied why some advocacy organizations are treated as authoritative voices in U.S. congressional floor speeches while others are mentioned only in passing. Using NLP text classification and multilevel regression models, I analyzed prominence patterns across the 114th and 115th Congress (2015–2019).

## Key Findings

- The top 1% of organizations account for ~31% of all prominent mentions. Prominence is heavily concentrated.
- Organizations using external lobbyists had significantly higher prominence odds (p = 0.001), contradicting the expectation that outsourcing advocacy signals weak leadership.
- Medium-salience policy areas show higher prominence than high-salience ones. Crowded issue spaces appear to dilute individual group visibility.
- Politician seniority has a small but significant negative effect on affording prominence, counter to the intuition that longer-serving members have stronger group relationships.

## Data

- ~78,000 Congressional Record documents (114th–115th Congress)
- 5,447 organizations from the Washington Representatives Study (2011)
- ~20,699 mention passages extracted, 19,165 in the analytical sample
- Full-sample design: includes organizations with zero mentions as baseline

## Methods

- Text classification: SVM with count vectorization (F1 = 0.79, accuracy ~81%, ROC AUC 0.72)
- Statistical models: Generalized Linear Mixed-Effects Models (GLMMs) in R (lme4)
- Random effects: Organization and Policy Area (crossed)
- Training data: 1,000 hand-coded passages following Fraussen et al. (2018) coding scheme

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

A modernized re-implementation of this pipeline with improved extraction and classification (F1 = 0.91, Cohen's kappa = 0.82) is available at [ThesisPipelineRework](https://github.com/kmazurek95/ThesisPipelineRework).

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata.

## License

MIT License - see [LICENSE](LICENSE).

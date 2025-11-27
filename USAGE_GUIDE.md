# Usage Guide: Interest Group Prominence Analysis

**Quick Start Guide for Researchers, Students, and Collaborators**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Overview](#repository-overview)
3. [Installation & Setup](#installation--setup)
4. [Working with the Data](#working-with-the-data)
5. [Running the Analysis](#running-the-analysis)
6. [Understanding the Results](#understanding-the-results)
7. [Extending the Research](#extending-the-research)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Want to dive right in?** Here's the 5-minute version:

```bash
# 1. Clone the repository
git clone https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis.git
cd MastersThesis_InterestGroupAnalysis

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Open the main analysis notebook
jupyter notebook analysis/01_Exploratory_Prominence_Analysis.ipynb
```

**That's it!** The notebook will guide you through loading and exploring the data.

---

## Repository Overview

### What This Repository Contains

This repository documents a **Master's thesis project** analyzing why some advocacy groups receive more recognition from U.S. politicians than others. It includes:

- **Complete dataset** (`multi_level_data/`) - 77,000+ Congressional Record documents from 2014-2018
- **Analysis code** (`analysis/`) - Jupyter notebooks and R scripts to reproduce findings
- **Original pipeline** (`legacy/`) - Historical code from the 2023 thesis research
- **Documentation** - Thesis PDF, technical reports, and data dictionaries

### Repository Philosophy

This project prioritizes **transparency over perfection**:

✅ **What's reproducible**: All statistical analyses and findings (from the processed dataset)
⚠️ **What's not**: The full data collection pipeline (due to manual steps and missing intermediates)
📚 **What's preserved**: All original code and decisions for historical reference

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** (for data analysis)
- **R 4.0+** (for multilevel statistical models)
- **Jupyter Notebook** (for interactive analysis)
- **Git** (for version control)

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis.git
cd MastersThesis_InterestGroupAnalysis
```

#### 2. Set Up Python Environment

**Option A: Using venv (recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using conda**

```bash
# Create conda environment
conda create -n thesis-analysis python=3.9
conda activate thesis-analysis

# Install dependencies
pip install -r requirements.txt
```

#### 3. Set Up R Environment

Open R or RStudio and run:

```r
# Install required packages
install.packages(c(
    "lme4",          # Mixed-effects models
    "ggplot2",       # Visualization
    "broom.mixed",   # Model tidying
    "arrow",         # Reading Parquet files
    "dplyr",         # Data manipulation
    "knitr",         # R Markdown
    "rmarkdown"      # Document generation
))
```

#### 4. Verify Installation

```bash
# Check Python
python --version
jupyter --version

# Check R (in R console)
R.version.string
```

---

## Working with the Data

### Understanding the Dataset

The **`multi_level_data`** dataset is the foundation of all analyses. It contains:

- **77,000+ Congressional Record documents** (2014-2018)
- **Interest group mentions** with prominence labels
- **Speaker metadata** (seniority, party, state)
- **Group-level variables** (lobbying expenditure, policy focus)
- **Policy-level variables** (salience scores, topic categories)

### Dataset Location

```
data/
└── multi_level_data/
    ├── README.md                               # Dataset documentation
    ├── df_interest_group_prominence_FINAL.csv  # Main analysis dataset
    ├── level1_FINAL.csv                        # Level 1 (mentions) data
    └── multi_level_data.csv                    # Complete hierarchical data
```

### Loading the Data

#### In Python

```python
import pandas as pd

# Load the main analysis dataset
df = pd.read_csv('data/multi_level_data/df_interest_group_prominence_FINAL.csv')

# Explore
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check prominence distribution
print(f"\nProminence distribution:")
print(df['prominent'].value_counts())
```

#### In R

```r
library(readr)
library(dplyr)

# Load the dataset
df <- read_csv("data/multi_level_data/df_interest_group_prominence_FINAL.csv")

# Explore
dim(df)
str(df)
summary(df$prominent)

# Preview
head(df)
```

### Key Variables Explained

| Variable | Description | Example Values |
|----------|-------------|----------------|
| `prominent` | Binary label: 1 = prominent mention, 0 = routine | 0, 1 |
| `interest_group_name` | Standardized name of advocacy organization | "American Medical Association" |
| `lobbying_expenditure` | Total lobbying spending (USD) | 1000000 |
| `policy_area` | Primary policy domain | "Healthcare", "Environment" |
| `policy_salience` | Google Trends salience score | 0.45 |
| `speaker_seniority` | Years in Congress | 12 |
| `speaker_party` | Political party | "D", "R", "I" |

**For complete variable definitions**, see [data/multi_level_data/README.md](data/multi_level_data/README.md)

---

## Running the Analysis

### Analysis Workflow

The analysis is organized into **three sequential notebooks/scripts**:

1. **[01_Exploratory_Prominence_Analysis.ipynb](analysis/01_Exploratory_Prominence_Analysis.ipynb)** ⭐ **START HERE**
   - Load and explore the dataset
   - Visualize patterns in interest group prominence
   - Descriptive statistics and data quality checks
   - **This is the centerpiece** of the portfolio analysis

2. **[02_Statistical_Models.ipynb](analysis/02_Statistical_Models.ipynb)**
   - Fit multilevel regression models
   - Test hypotheses about prominence drivers
   - Model diagnostics and validation

3. **[03_Multilevel_Models.Rmd](analysis/03_Multilevel_Models.Rmd)**
   - Advanced generalized linear mixed-effects models (GLMM)
   - Reproduce thesis findings in R
   - Generate publication-ready tables

### Running Jupyter Notebooks

```bash
# Navigate to analysis directory
cd analysis

# Start Jupyter Notebook
jupyter notebook

# Or open a specific notebook
jupyter notebook 01_Exploratory_Prominence_Analysis.ipynb
```

**In the browser:**
1. Click on the notebook file
2. Run cells sequentially using `Shift + Enter`
3. Explore outputs and modify code as needed

### Running R Markdown Scripts

```bash
# Navigate to analysis directory
cd analysis

# Render the R Markdown document
Rscript -e "rmarkdown::render('03_Multilevel_Models.Rmd')"
```

Or open in **RStudio**:
1. Open `03_Multilevel_Models.Rmd`
2. Click "Knit" button
3. Choose output format (HTML, PDF, or Word)

### Expected Outputs

After running all analyses, you should have:

**Figures** (saved to `output/figures/`):
- Prominence rates by policy area
- Lobbying expenditure vs. prominence scatter plots
- Temporal trends in group mentions
- Model coefficient plots

**Tables** (saved to `output/tables/`):
- Descriptive statistics
- Regression model results
- Robustness checks

---

## Understanding the Results

### Key Findings from the Thesis

1. **Lobbying Expenditure Matters**
   - Groups that spend more on lobbying receive more prominent mentions
   - Effect is positive and statistically significant

2. **Broad Policy Engagement Increases Visibility**
   - Groups working across multiple policy areas get more recognition
   - Specialization may limit prominence

3. **Medium-Salience Issues Show Highest Prominence**
   - Counter-intuitively, medium-salience policy areas show more group visibility
   - High-salience issues may crowd out individual group recognition

4. **Seniority Effects Are Weak or Negative**
   - Contrary to expectations, senior politicians don't always give more prominent mentions
   - May reflect different communication strategies

### Interpreting the Models

The core statistical model is a **Generalized Linear Mixed-Effects Model (GLMM)**:

```
Prominence ~ Lobbying Expenditure + Policy Breadth + Membership Size +
             Speaker Seniority + Speaker Party + Policy Salience +
             (1 | Policy Area) + (1 | Interest Group)
```

**Fixed effects**: Predict how variables affect prominence across all groups
**Random effects**: Account for clustering within policy areas and interest groups

**Reading the output:**
- **Coefficient > 0**: Positive association with prominence
- **p < 0.05**: Statistically significant relationship
- **Confidence intervals**: Uncertainty range for estimates

---

## Extending the Research

### Want to Build on This Work?

Here are some ideas for extensions:

#### 1. Test New Hypotheses

```python
# Example: Does committee membership affect prominence?
# Add committee data and re-run models
df['on_relevant_committee'] = ...  # Add your variable
```

#### 2. Analyze Different Time Periods

```python
# Filter to a specific Congress
df_114 = df[df['congress'] == 114]
df_115 = df[df['congress'] == 115]

# Compare results
```

#### 3. Try Alternative Models

```r
# Random slopes model
model_alt <- glmer(
    prominent ~ lobbying_expenditure +
                (1 + lobbying_expenditure | policy_area),
    data = df,
    family = binomial
)
```

#### 4. Visualize New Patterns

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Custom visualization
sns.scatterplot(data=df, x='policy_salience', y='prominent',
                hue='policy_area')
plt.title('Your Custom Analysis')
plt.show()
```

### Contributing Back

If you extend this research:

1. **Fork** the repository
2. **Create a branch** for your analysis (`git checkout -b new-analysis`)
3. **Add your code** to the `analysis/` folder
4. **Document** your approach in a new notebook
5. **Submit a pull request** with a clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Troubleshooting

### Common Issues

#### Dataset File Not Found

**Error**: `FileNotFoundError: data/multi_level_data/...csv`

**Solution**: Ensure you're running code from the repository root directory:

```bash
# Check current directory
pwd

# If not in repository root, navigate there
cd /path/to/MastersThesis_InterestGroupAnalysis
```

#### Missing Python Packages

**Error**: `ModuleNotFoundError: No module named 'pandas'`

**Solution**: Install dependencies:

```bash
pip install -r requirements.txt
```

#### R Package Installation Fails

**Error**: Package installation errors in R

**Solution**: Try installing packages individually:

```r
# Install one at a time
install.packages("lme4")
install.packages("ggplot2")
# ... etc.
```

For binary packages (Windows):
```r
install.packages("lme4", type = "win.binary")
```

#### Models Won't Converge

**Error**: `Model failed to converge`

**Solution**: Try these approaches:

```r
# Scale continuous variables
df$lobbying_scaled <- scale(df$lobbying_expenditure)

# Simpler random effects
model <- glmer(prominent ~ ... + (1 | policy_area),
               data = df, family = binomial)

# Different optimizer
model <- glmer(..., control = glmerControl(optimizer = "bobyqa"))
```

#### Jupyter Notebook Won't Start

**Error**: `Jupyter command not found`

**Solution**: Ensure Jupyter is installed in your environment:

```bash
# Activate virtual environment first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Jupyter
pip install jupyter

# Try again
jupyter notebook
```

#### Data Too Large for Memory

**Error**: `MemoryError` when loading data

**Solution**: Use chunked reading or Dask:

```python
# Read in chunks
chunks = pd.read_csv('data/multi_level_data/multi_level_data.csv',
                      chunksize=10000)

# Or use Dask for larger-than-memory data
import dask.dataframe as dd
df = dd.read_csv('data/multi_level_data/multi_level_data.csv')
```

### Getting Help

If you encounter issues not covered here:

1. **Check existing documentation**:
   - [README.md](README.md) - Main repository overview
   - [data/multi_level_data/README.md](data/multi_level_data/README.md) - Dataset docs
   - [analysis/README.md](analysis/README.md) - Analysis instructions

2. **Search GitHub issues**: Check if others have had similar problems

3. **Open a new issue**: [Create an issue](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues) with:
   - Clear description of the problem
   - Error messages (copy-paste full output)
   - Your environment (OS, Python version, etc.)
   - Steps to reproduce

4. **Contact the author**:
   - Email: kalebmazurek@gmail.com
   - GitHub: [@kmazurek95](https://github.com/kmazurek95)

---

## Additional Resources

### Essential Reading

- **Thesis PDF**: [legacy/5. Visualization and Reporting/Thesis_UvA_Kaleb_Mazurek.pdf](legacy/5.%20Visualization%20and%20Reporting/)
- **Technical Report**: [legacy/5. Visualization and Reporting/Technical Report MA Thesis.pdf](legacy/5.%20Visualization%20and%20Reporting/)
- **Data Dictionary**: [data/multi_level_data/README.md](data/multi_level_data/README.md)

### External Resources

**Learning Multilevel Modeling**:
- [GLMM FAQ](https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html) - Ben Bolker's comprehensive guide
- [lme4 Documentation](https://cran.r-project.org/web/packages/lme4/lme4.pdf)

**Congressional Data**:
- [GovInfo API](https://api.govinfo.gov/docs/) - Congressional Record source
- [Congress.gov API](https://api.congress.gov/) - Legislative data

**Python/R for Political Science**:
- [Computational Social Science](https://sicss.io/) - SICSS resources
- [Quantitative Politics with R](http://qpolr.com/)

---

## Citation

If you use this repository in your research, please cite:

### Thesis

```bibtex
@mastersthesis{mazurek2023prominence,
  author  = {Mazurek, Kaleb},
  title   = {Beyond Policy Influence: A Deeper Dive into the Factors
             Driving Advocacy Group Prominence},
  school  = {University of Amsterdam},
  year    = {2023},
  type    = {Master's Thesis},
  url     = {https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis}
}
```

### Dataset

```bibtex
@dataset{mazurek2024multilevel,
  author  = {Mazurek, Kaleb},
  title   = {Multi-Level Interest Group Prominence Dataset:
             U.S. Congressional Record 2014-2018},
  year    = {2024},
  version = {1.0},
  url     = {https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis}
}
```

---

## License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

**University of Amsterdam** - Political Science Department
**U.S. Government Publishing Office** - GovInfo API access
**Library of Congress** - Congress.gov API

---

**Last Updated**: November 26, 2024
**Version**: 2.0.0
**Maintainer**: Kaleb Mazurek (kalebmazurek@gmail.com)

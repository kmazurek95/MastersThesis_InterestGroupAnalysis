# README: Modeling Prominence in Interest Group Mentions

## Overview

This repository contains scripts and analyses for modeling the prominence of interest group mentions in legislative debates. The research explores factors that influence prominence, including issue areas, group characteristics, and politician attributes. The dataset operationalizes key variables to analyze these dynamics and applies Generalized Linear Mixed-Effects Models (GLMMs) to uncover significant predictors.

## Purpose

The project investigates the interplay of three primary factors:
- **Interest Groups**: Attributes such as age, membership status, and lobbying expenditure.
- **Issue Areas**: Contextualized through policy domains and public salience.
- **Speakers (Politicians)**: Attributes such as policy-domain overlap, seniority, and election-year status.

## Repository Structure

```bash
.
├── data/                  # Raw and preprocessed datasets
├── scripts/               # Python and R scripts for data preprocessing and modeling
│   ├── model_policy_salience.R               # R: Models salience-related factors
│   ├── model_group_politician.R              # R: Models politician-group linkages
│   ├── model_group_characteristics.R         # R: Models group-level attributes
│   ├── DataProcessingAndRegression.py        # Python: Preprocessing, cleaning, and experimental regression analysis
│   ├── InterestGroupAnalysisPipeline.py      # Python: Interest group data integration and validation pipeline
├── results/               # Outputs (e.g., tables, plots, metrics)
├── README.md              # Project documentation
```

---

## Modeling Approach

The analysis employs **Generalized Linear Mixed-Effects Models (GLMMs)**, allowing for:
- Binary outcomes (prominent vs. non-prominent mentions).
- Separation of fixed and random effects.
- Accounting for hierarchical structures in the data (mentions nested within interest groups and issue areas).

### Key Models:
#### 1. **Policy Salience Models**:
- Explores how public salience of policy areas impacts the likelihood of prominent mentions.
- Incorporates salience categories (low, medium, high) derived from Google Trends data.

#### 2. **Group-Politician Linkage Models**:
- Evaluates how politician attributes (e.g., bill sponsorship, seniority) influence prominence.
- Includes random effects for policy areas and interest groups.

#### 3. **Group Characteristics Models**:
- Investigates how group attributes (e.g., lobbying expenditure, age, policy scope) predict prominence.
- Highlights the significance of broad policy engagement.

---

## Description of Python Scripts

### `DataProcessingAndRegression.py`

**Purpose:**  
Handles comprehensive data preprocessing and regression analysis, including cleaning, visualization, and modeling.

**Key Features:**
- **Data Cleaning:** Replaces missing values and filters invalid rows.
- **Feature Engineering:**  
  - Processes date columns to compute derived variables like `YEARS_EXISTED`.
  - Normalizes salience metrics and creates ordinal variables for analysis.
- **Visualization:**  
  - Generates histograms and boxplots for raw and transformed data.
  - Provides insights into variable distributions before and after transformations.
- **Probit Regression:** Fits a Probit regression model to predict binary outcomes like prominence.
- **Data Aggregation:** Groups data by organizational ID and summarizes metrics like lobbying expenditure, years existed, and prominence counts.
- **Output:** Saves cleaned, transformed, and aggregated datasets to CSV files.

**Usage:**  
Run the script to preprocess datasets, visualize key variables, and perform regression modeling. Outputs include clean datasets, statistical summaries, and visualizations.

---

### `InterestGroupAnalysisPipeline.py`

**Purpose:**  
Focuses on advanced data integration, validation, and deduplication tasks for interest group analysis.

**Key Features:**
- **Duplicate Detection:**  
  - Uses MinHash and LSH Forest algorithms to identify and filter potential duplicate records.
  - Facilitates efficient identification of overlaps in variations and organization IDs.
- **Dataset Merging:** Combines multiple datasets based on a common key (e.g., `granuleId`) while applying filtering rules.
- **Data Validation:**  
  - Checks for duplicate or missing records in critical columns.
  - Ensures dataset integrity for downstream analyses.
- **Interest Group Filtering:** Removes specific interest group categories based on custom criteria.
- **Feature Engineering:** Adds a `congress` column by mapping dates to congressional sessions.
- **Output:** Saves validated and merged datasets for further analysis.

**Usage:**  
Run the script to clean, validate, and merge datasets, and prepare integrated data pipelines. Outputs include filtered datasets and enriched features like congressional mappings.

---

## Data & Variable Operationalization

### Variables
1. **Prominence**: Binary indicator (prominent vs. non-prominent mentions) derived using a classifier.
2. **Issue Area**: Categorized into 21 domains using the Comparative Agendas Project schema and committee/bill mapping.
3. **Public Salience**: Based on Google Trends data, categorized into low, medium, and high salience.
4. **Group Attributes**: Includes age, membership status, lobbying expenditure, and policy scope.
5. **Speaker Attributes**: Includes seniority, election-year status, and policy-domain overlap.

### Key Data Sources:
- **Congressional Records**: For mentions and associated metadata.
- **Google Trends**: To derive issue salience measures.
- **Washington Representative Studies**: For group-level attributes.
- **ProPublica API**: For politician characteristics.

---

## Results Summary

### Policy Salience:
- Medium salience policy areas significantly increase the likelihood of prominent mentions.
- High salience areas do not show significant effects.

### Group-Politician Linkage:
- Senior politicians are less likely to afford prominence to interest groups, contrary to expectations.
- Bill sponsorship and policy-area overlap show limited significance.

### Group Characteristics:
- Engagement with a broad range of policy areas enhances prominence significantly.
- Lobbying expenditure becomes significant when control variables are included.
- Organization age shows limited influence on prominence.

---

## How to Use the Repository

### Clone the Repository:
```bash
git clone https://github.com/username/repo-name.git
```

### Install Dependencies:

For Python scripts:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels datasketch tqdm
```

For R scripts:
```R
install.packages(c("lme4", "dplyr", "ggplot2", "broom.mixed", "kableExtra", "forcats"))
```

### Run Scripts:

1. **Python Scripts**:
   - `DataProcessingAndRegression.py`: Handles preprocessing, visualization, and regression modeling.
   - `InterestGroupAnalysisPipeline.py`: Integrates, validates, and deduplicates datasets.
2. **R Scripts**:
   - `model_policy_salience.R`
   - `model_group_politician.R`
   - `model_group_characteristics.R`

### Outputs:

Results are saved in the `results/` directory, including:
- Cleaned datasets.
- Statistical summaries.
- Visualizations.

---

## Limitations
1. **Data Completeness**: Certain observations lack policy context and were excluded from the analysis.
2. **Google Trends Data**: Represents public attention but may not perfectly align with salience.
3. **Temporal Constraints**: Some group-level attributes (e.g., lobbying expenditure) are treated as constant due to data availability.

---

## Citation
If using this repository for your research, please cite:
> Beyond Policy Influence: A Deeper Dive into the Factors Driving
Advocacy Group Prominence. University of Amsterdam. 2023.

---

## License
This repository is licensed under the MIT License. See `LICENSE` for details.

---

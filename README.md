# Masters Thesis: Interest Group Analysis

## Overview
This repository contains the complete workflow for my master's thesis, **"Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence,"** conducted at the University of Amsterdam. The research explores the dynamics of interest group prominence in legislative debates, leveraging advanced statistical methods, supervised learning, and integrated datasets.

Prominence, a key concept in this study, reflects the recognition and soft power afforded to advocacy organizations by policymakers. By analyzing legislative debates from the 114th and 115th U.S. Congress sessions, this study identifies the factors influencing prominence and challenges conventional notions about interest group success.

## Research Questions
1. Why do some politicians prioritize certain advocacy organizations?
2. What factors contribute to variations in prominence across groups and issues?

## Skills Demonstrated
This project demonstrates proficiency in the following technical and analytical skills:

- **Data Collection**:
  - Web scraping and API usage (e.g., GovInfo API, Wikipedia API).
  - Handling large legislative datasets (~77,000 documents).
- **Data Processing**:
  - Data cleaning and preprocessing using Python (`pandas`, `numpy`).
  - Feature engineering and transformation for machine learning pipelines.
- **Machine Learning**:
  - Implementation of supervised learning models (e.g., Support Vector Machines) for text classification.
  - Hyperparameter tuning and evaluation of models using metrics like accuracy, precision, recall, and F1-score.
- **Statistical Analysis**:
  - Generalized Linear Mixed-Effects Modeling using R (`lme4`, `broom.mixed`).
  - Multilevel regression analysis to explore policy, group, and politician-level factors.
- **Data Integration**:
  - Combining structured and unstructured data into a unified dataset.
  - Feature extraction from textual data (e.g., legislative mentions).
- **Visualization and Reporting**:
  - Data visualization using Python (`seaborn`, `matplotlib`) and R (`ggplot2`).
  - Reporting insights in a structured, research-driven format using R Markdown and Python scripts.
- **Project Management**:
  - Organizing workflows and managing a multi-stage analysis pipeline.
  - Documenting and structuring the repository for reproducibility.

## Repository Structure
```bash
.
├── LICENSE                                # License details for this repository
├── README.md                              # High-level overview of the repository
├── MasterThesisUniversityOfAmsterdam/     # Main repository directory
│   ├── 1. Data Collection/                # Scripts and data for collecting raw legislative data
│   ├── 2. Data Processing/                # Scripts for cleaning and preparing data for analysis
│   ├── 3. Supervised Learning Classifiers/ # Classifiers for predicting prominence using supervised methods
│   ├── 4. Integrated Dataset and Analysis/ # Advanced data analysis and modeling pipeline
│   │   ├── DataProcessingAndRegression.py    # Python script for regression analysis and feature engineering
│   │   ├── InterestGroupAnalysisPipeline.py  # Comprehensive pipeline for interest group-level analysis
│   ├── 5. Visualization and Reporting/    # Scripts for visualizing results and generating reports
```

## Methodologies
### Data Sources
- **Congressional Record**: Over 77,000 legislative documents analyzed for mentions of interest groups.
- **Washington Representative Study**: Provides group-level variables (e.g., age, membership status).
- **Google Trends**: Measures public salience of policy areas.

### Analytical Framework
1. **Supervised Machine Learning**:
   - A Support Vector Machine classifier identified prominent mentions within congressional debates (~81% accuracy).
2. **Generalized Linear Mixed-Effects Models**:
   - Explored how factors at the policy, group, and politician levels influence prominence.
3. **Data Integration**:
   - Combined legislative data, interest group characteristics, and public salience metrics.

## Key Findings
- Senior politicians are less likely to afford prominence to interest groups.
- High or medium saliency policy areas do not guarantee greater prominence for advocacy groups.
- Advocacy organizations engaging across diverse policy areas are more likely to gain prominence.
- External lobbying expenditure influences prominence only when controlling for other variables.

## Usage
### Clone the Repository
```bash
git clone https://github.com/username/MastersThesis_InterestGroupAnalysis.git
```

### Navigate to Folders
- Each folder contains scripts

and data relevant to a specific stage of the analysis.
- Review the README files within subfolders for script-specific instructions.

### Run Scripts
- Follow instructions in the subfolder-specific README files to execute scripts.

## Requirements
- **R**: Required libraries include `lme4`, `ggplot2`, `dplyr`, `broom.mixed`, etc.
- **Python**: Required libraries include `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `seaborn`, etc.

## Contribution
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## Citation
If using this repository in your research, please cite:

Mazurek, Kaleb. *Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence*. University of Amsterdam, 2023.

## License
This repository is licensed under the MIT License. See LICENSE for details.

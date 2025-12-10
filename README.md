# Interest Group Analysis: Legislative Prominence Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research: Completed](https://img.shields.io/badge/research-completed-green.svg)]()

> **Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence**
>
> *Master's Thesis, University of Amsterdam (2023)*
> *Author: Kaleb Mazurek*

---

## Project Overview

This repository contains research examining **why some advocacy organizations receive more recognition from politicians than others** in U.S. Congressional debates. Using 77,000+ legislative documents from the 114th and 115th Congress (2014-2018), this study combines:

- **Natural Language Processing**: Extracting interest group mentions from legislative text
- **Machine Learning**: Supervised classification of "prominent" vs. routine mentions (81% accuracy)
- **Statistical Modeling**: Multilevel regression analysis of prominence drivers
- **Data Integration**: Combining congressional records, lobbying data, and policy salience metrics

### Key Findings

- **Lobbying expenditure** and **broad policy engagement** strongly predict prominence
- **Medium-salience policy areas** show higher advocacy visibility than high-salience areas
- **Politician seniority** shows unexpected negative or null effects on prominence patterns

---

## Repository Purpose

This repository serves two main purposes:

### 1. **Legacy Code Archive**

The `/legacy/` folder contains the **original thesis code** exactly as it existed during the 2023 research. This code is preserved for:
- Historical reference and transparency
- Understanding the original methodology
- Learning about the research process

**Important**: The legacy code is **not fully reproducible** due to manual steps, missing intermediate data, and ad-hoc processing. See [`legacy/README.md`](./legacy/README.md) for details.

### 2. **Reproducible Dataset**

The **`multi_level_data`** dataset provides a **validated, analysis-ready starting point** for reproducing and extending the thesis findings.

**This dataset is the ONLY reproducible component** of this repository.

---

## Repository Structure

```
.
├── README.md                      # This file
├── USAGE_GUIDE.md                 # Detailed usage instructions - START HERE
├── CONTRIBUTING.md                # How to contribute to this project
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
│
├── legacy/                        # ORIGINAL THESIS CODE (2023)
│   ├── README.md                  # Why legacy code is not reproducible
│   ├── 1. Data Collection/        # Original GovInfo API scripts
│   ├── 2. Data Proccessing/       # Original processing scripts
│   ├── 3. Supervised Learning Classifiers/
│   ├── 4. Integrated Dataset and Analysis/
│   └── 5. Visualization and Reporting/
│       ├── Thesis_UvA_Kaleb_Mazurek.pdf
│       └── Technical Report MA Thesis.pdf
│
├── data/                          # DATA
│   └── multi_level_data/          # THE REPRODUCIBILITY ANCHOR
│       ├── README.md              # Complete dataset documentation
│       ├── df_interest_group_prominence_FINAL.csv  # Main analysis dataset
│       ├── level1_FINAL.csv       # Level 1 (mentions) data
│       └── multi_level_data.csv   # Complete hierarchical data
│
├── analysis/                      # REPRODUCIBLE ANALYSIS
│   ├── README.md                  # How to reproduce thesis findings
│   ├── 01_Exploratory_Prominence_Analysis.ipynb  # MAIN ANALYSIS
│   ├── 02_Statistical_Models.ipynb               # Statistical modeling
│   └── 03_Multilevel_Models.Rmd                  # R-based GLMM models
│
├── output/                        # ANALYSIS OUTPUTS
│   ├── figures/                   # Publication-ready plots
│   └── tables/                    # Regression tables
│
└── docs/                          # DOCUMENTATION
    └── README.md                  # Documentation overview
```

---

## Quick Start: Reproducing Thesis Results

### For First-Time Users

** See [USAGE_GUIDE.md](USAGE_GUIDE.md) for comprehensive instructions**

### Fast Track (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis.git
cd MastersThesis_InterestGroupAnalysis

# 2. Install Python dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Open the main analysis notebook
jupyter notebook analysis/01_Exploratory_Prominence_Analysis.ipynb
```

### For R Users

```r
# Install R packages
install.packages(c("lme4", "ggplot2", "broom.mixed", "readr", "dplyr", "knitr"))

# Open the R Markdown file
# analysis/03_Multilevel_Models.Rmd
```

### Dataset Access

The data files are available in [data/multi_level_data/](./data/multi_level_data/):
- `df_interest_group_prominence_FINAL.csv` - Main analysis dataset
- `level1_FINAL.csv` - Level 1 (mentions) data
- `multi_level_data.csv` - Complete hierarchical data

**Load in Python:**
```python
import pandas as pd
data = pd.read_csv('data/multi_level_data/df_interest_group_prominence_FINAL.csv')
```

**Load in R:**
```r
library(readr)
data <- read_csv("data/multi_level_data/df_interest_group_prominence_FINAL.csv")
```

---

## 📊 The `multi_level_data` Dataset

### What Is It?

The **`multi_level_data`** dataset is a **cleaned, validated, hierarchical dataset** that contains:

- **77,000+ Congressional Record documents** (2014-2018)
- **Processed paragraphs** with speaker assignments
- **Interest group mentions** with prominence labels
- **Machine learning predictions** (SVM classifier outputs)
- **Speaker metadata** (seniority, party, state, committee membership)
- **Group-level variables** (lobbying expenditure, policy focus, membership)
- **Policy-level variables** (salience, topic categories)

### Why Start Here?

The original thesis pipeline had **manual steps** and **missing intermediate data** that prevent full reproduction from raw Congressional Record data. The `multi_level_data` dataset provides:

**Complete reproducibility** for all thesis findings
**Validated ground truth** - all variables checked and documented
**Analysis-ready format** - no preprocessing needed
**Comprehensive documentation** - data dictionary with all variables

### Structure

```
Hierarchical levels:
  Policy Areas (Level 3)
      └── Interest Groups (Level 2)
          └── Mentions (Level 1)
              └── Paragraphs
                  └── Congressional Record Documents
```

**See**: [`data/multi_level_data/README.md`](./data/multi_level_data/README.md) for complete documentation.

---

## Key Concepts

### What Is "Prominence"?

**Prominence** refers to the **depth and quality of recognition** an advocacy organization receives in legislative debates. It goes beyond simple mentions to capture:

- **Substantive engagement** with group positions
- **Acknowledgment of expertise** or authority
- **Discussion of group activities** or recommendations
- **Soft power** and influence signals

**Example of prominent mention**:
> "The American Medical Association has conducted extensive research on this issue, and their recommendations should guide our policy decisions."

**Example of routine mention**:
> "We received letters from the AMA and several other organizations."

---

## Research Methodology

### 1. Data Collection (2014-2018)
- **Source**: GovInfo API (Congressional Record)
- **Volume**: 77,000+ legislative documents
- **Metadata**: Congress.gov API (member profiles, bill data)
- **External Data**: Washington Representatives Study, Google Trends

### 2. Natural Language Processing
- **Speaker Assignment**: Rule-based heuristics (~85-90% accuracy)
- **Interest Group Extraction**: Pattern matching + Named Entity Recognition
- **Duplicate Detection**: TF-IDF cosine similarity filtering

### 3. Supervised Learning
- **Training Data**: 2,000 hand-labeled mentions
- **Models**: Support Vector Machines, Naive Bayes, Random Forest
- **Best Performance**: SVM with 81% accuracy, 0.79 F1-score

### 4. Statistical Modeling
- **Framework**: Generalized Linear Mixed-Effects Models (GLMM)
- **Dependent Variable**: Prominence (binary, from classifier)
- **Levels**: Policy area (Level 3), Interest group (Level 2), Mention (Level 1)
- **Software**: R with `lme4`, `broom.mixed`, `ggplot2`

---

## Documentation

### Essential Reading

1. **[data/multi_level_data/README.md](./data/multi_level_data/README.md)** - Complete dataset documentation
2. **[docs/data_dictionary.md](./docs/data_dictionary.md)** - All variable definitions (coming soon)
3. **[docs/reproducibility_guide.md](./docs/reproducibility_guide.md)** - Step-by-step reproduction (coming soon)
4. **[legacy/README.md](./legacy/README.md)** - Why original code is not reproducible

### Thesis Documents

- **[legacy/5. Visualization and Reporting/Thesis_UvA_Kaleb_Mazurek.pdf](./legacy/5.%20Visualization%20and%20Reporting/Thesis_UvA_Kaleb_Mazurek.pdf)** - Full thesis manuscript
- **[legacy/5. Visualization and Reporting/Technical Report MA Thesis.pdf](./legacy/5.%20Visualization%20and%20Reporting/Technical%20Report%20MA%20Thesis.pdf)** - Technical methodology

---

## Using This Repository

### For Researchers Wanting to Reproduce Results

1. **Access** the `multi_level_data` dataset
2. **Load** the data in Python or R
3. **Run** the analysis notebooks in `/analysis/`
4. **Compare** your results with the thesis findings

### For Researchers Wanting to Extend the Analysis

1. **Start** with the `multi_level_data` dataset
2. **Explore** alternative models or variables
3. **Test** new hypotheses about prominence
4. **Contribute** your findings back (via pull request)

### For Researchers Wanting to Understand the Methodology

1. **Read** the thesis PDF for conceptual framework
2. **Review** the legacy code to see original implementation
3. **Consult** the data dictionary for variable definitions
4. **Examine** the technical report for detailed methods

### For Teachers/Students Learning Computational Social Science

1. **Use** as a case study in research transparency
2. **Discuss** the tradeoffs between exploratory research and reproducibility
3. **Practice** multilevel modeling with real political science data
4. **Learn** from both successes and documented limitations

---

## What This Repository Does NOT Provide

**Raw Congressional Record data** - Too large, must be collected via API
**Full pipeline from scratch** - Manual steps and missing data prevent this
**Data collection scripts that work out-of-the-box** - Legacy code requires adaptation
**New data for recent Congresses** - Covers 2014-2018 only

**What it DOES provide:**
**Complete reproducibility from the validated dataset**
**Transparent documentation of limitations**
**All code and decisions from original research**
**Analysis-ready data for extensions**

---

## Citation

If you use this data or methodology in your research, please cite:

### Thesis Citation

```bibtex
@mastersthesis{mazurek2023prominence,
  author  = {Mazurek, Kaleb},
  title   = {Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence},
  school  = {University of Amsterdam},
  year    = {2023},
  type    = {Master's Thesis},
  url     = {https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis}
}
```

### Dataset Citation

```bibtex
@dataset{mazurek2024multilevel,
  author  = {Mazurek, Kaleb},
  title   = {Multi-Level Interest Group Prominence Dataset: U.S. Congressional Record 2014-2018},
  year    = {2024},
  version = {1.0},
  url     = {https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis}
}
```

---

## 📧 Contact

**Kaleb Mazurek**
- Email: kalebmazurek@gmail.com
- GitHub: [@kmazurek95](https://github.com/kmazurek95)

For questions about:
- **Dataset access**: See [data/multi_level_data/README.md](./data/multi_level_data/README.md)
- **Reproducing results**: See [docs/reproducibility_guide.md](./docs/reproducibility_guide.md)
- **Original methodology**: See [legacy/README.md](./legacy/README.md)
- **Technical issues**: [Open a GitHub issue](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Data

- **Congressional Record**: Public domain (U.S. Government work)
- **Washington Representatives Study**: Used with permission
- **Google Trends**: Accessed via public API

---

## Acknowledgments

### Institutions
- **University of Amsterdam** - Political Science Department
- **U.S. Government Publishing Office** - GovInfo API access
- **Library of Congress** - Congress.gov API

### Philosophy

This repository demonstrates that **research transparency** means acknowledging both successes and limitations:

> "Science is a process, not a product. Perfect reproducibility from the first step is rarely achievable in exploratory research. What matters is providing a validated starting point for future work."

---

## Project Statistics

- **Original Data Collection**: 77,000+ Congressional Record documents
- **Time Period**: 2014-2018 (114th & 115th Congress)
- **Interest Group Mentions**: ~25,000 extracted mentions
- **Hand-Labeled Training Data**: 2,000 mentions
- **Classification Accuracy**: 81% (SVM model)
- **Statistical Models**: 3 levels (policy/group/mention)
- **Reproducible from**: `multi_level_data` dataset (validated)

---

**Last Updated**: November 25, 2024
**Version**: 2.0.0

---

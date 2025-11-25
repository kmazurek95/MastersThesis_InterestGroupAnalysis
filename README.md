# Interest Group Analysis: Legislative Prominence Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research Status: Modernizing](https://img.shields.io/badge/status-modernizing-orange.svg)]()

> **Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence**
>
> *Master's Thesis, University of Amsterdam (2023)*
> *Author: Kaleb Mazurek*

---

## 🎯 Project Overview

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

## 📣 Repository Status: Modernization in Progress

### What's New (November 2024)

This repository is undergoing a **comprehensive modernization** to transform the original thesis code into a **fully reproducible research pipeline**.

#### ✅ Completed
- **Legacy code archived** with full documentation of limitations
- **New repository structure** designed for reproducibility
- **Ground truth dataset created**: `multi_level_data` as the reproducibility anchor

#### 🚧 In Progress
- **Modular pipeline development**: Rebuilding from raw data to analysis
- **Comprehensive documentation**: Data dictionaries, API guides, reproducibility notes
- **Best practices implementation**: Version control, testing, containerization

#### 📋 Planned
- **Extension to new Congress sessions** (2024-2025)
- **Automated validation and testing**
- **Interactive data exploration tools**

👉 **See [ROADMAP.md](./ROADMAP.md) for the full development timeline**

---

## 📂 Repository Structure

```
.
├── README.md                      # This file
├── LICENSE                        # MIT License
├── MIGRATION_PLAN.md              # Detailed transition strategy
├── ROADMAP.md                     # Future development timeline
├── CITATION.cff                   # Citation metadata
│
├── legacy/                        # ⚠️ ORIGINAL THESIS CODE (2023)
│   ├── README.md                  # Why legacy code is not reproducible
│   ├── 1. Data Collection/        # Original GovInfo API scripts
│   ├── 2. Data Proccessing/       # Original processing scripts
│   ├── 3. Supervised Learning Classifiers/
│   ├── 4. Integrated Dataset and Analysis/
│   └── 5. Visualization and Reporting/
│       ├── Thesis_UvA_Kaleb_Mazurek.pdf
│       └── Technical Report MA Thesis.pdf
│
├── data/                          # 📊 DATA ORGANIZATION
│   ├── README.md                  # Data documentation and versioning
│   ├── raw/                       # Raw API outputs (not in git)
│   ├── processed/                 # Cleaned, structured datasets
│   │   └── multi_level_data/      # 🎯 REPRODUCIBILITY ANCHOR
│   ├── interim/                   # Intermediate processing outputs
│   └── external/                  # Third-party datasets
│
├── pipeline/                      # 🔬 MODERN REPRODUCIBLE PIPELINE
│   ├── README.md                  # Pipeline documentation
│   ├── 01_data_collection/        # Modular data collection scripts
│   ├── 02_data_processing/        # Validated processing functions
│   ├── 03_machine_learning/       # Classifier training and evaluation
│   ├── 04_analysis/               # Statistical modeling
│   └── utils/                     # Shared helper functions
│
├── docs/                          # 📚 COMPREHENSIVE DOCUMENTATION
│   ├── data_dictionary.md         # Variable definitions
│   ├── reproducibility_notes.md   # Step-by-step reproduction guide
│   ├── multi_level_data_specification.md
│   └── api_setup_guide.md         # GovInfo/Congress API setup
│
├── outputs/                       # 📈 ANALYSIS OUTPUTS
│   ├── figures/                   # Publication-ready plots
│   ├── tables/                    # Regression tables and summaries
│   └── models/                    # Trained model artifacts
│
├── notebooks/                     # 📓 EXPLORATORY ANALYSIS
│   └── README.md                  # Notebook documentation
│
└── tests/                         # ✅ UNIT AND INTEGRATION TESTS
    └── README.md                  # Testing documentation
```

---

## 🚀 Quick Start

### For Researchers Using Existing Data

If you want to **reproduce the thesis analysis** using the validated dataset:

```bash
# 1. Clone the repository
git clone https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis.git
cd MastersThesis_InterestGroupAnalysis

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Access the ground truth dataset
# See data/processed/multi_level_data/README.md for details

# 4. Run analysis notebooks
jupyter notebook notebooks/
```

### For Researchers Collecting New Data

If you want to **collect new Congressional Record data** for recent sessions:

```bash
# 1. Set up API credentials
# See docs/api_setup_guide.md

# 2. Run data collection pipeline
cd pipeline/01_data_collection
python fetch_congressional_records.py --congress 118 --year 2024

# 3. Follow processing steps
# See pipeline/README.md for full workflow
```

---

## 📊 The `multi_level_data` Dataset

### What Is It?

The **`multi_level_data`** file is a **cleaned, validated, hierarchical dataset** that serves as the **reproducibility anchor** for this project. It contains:

- **77,000+ Congressional Record documents** (2014-2018)
- **Processed paragraphs** with speaker assignments
- **Interest group mentions** with prominence labels
- **Machine learning predictions** (SVM classifier outputs)
- **Speaker metadata** (seniority, party, state, committee membership)
- **Group-level variables** (lobbying expenditure, policy focus, membership)
- **Policy-level variables** (salience, topic categories)

### Why Use It?

The original thesis pipeline had **manual steps** and **missing intermediate data** that prevent full reproduction. The `multi_level_data` dataset provides:

✅ **Validated ground truth** for all thesis findings
✅ **Complete variable documentation** with data dictionary
✅ **Consistent structure** for reproducible analysis
✅ **Version control** with documented data provenance

### How to Use It

See **[docs/multi_level_data_specification.md](./docs/multi_level_data_specification.md)** for:

- Variable definitions and data types
- Hierarchical structure (mentions nested in paragraphs in documents)
- Missing data patterns and handling
- Example analysis code

---

## 🔬 Research Methodology

### 1. Data Collection
- **Source**: GovInfo API (Congressional Record, 2014-2018)
- **Volume**: 77,000+ legislative documents
- **Metadata**: Congress.gov API (member profiles, bill data)
- **External Data**: Washington Representatives Study, Google Trends

### 2. Natural Language Processing
- **Speaker Assignment**: Rule-based heuristics with manual validation
- **Interest Group Extraction**: Pattern matching + Named Entity Recognition
- **Duplicate Detection**: TF-IDF cosine similarity filtering

### 3. Supervised Learning
- **Training Data**: 2,000 hand-labeled mentions
- **Models**: Support Vector Machines, Naive Bayes, Random Forest
- **Best Performance**: SVM with 81% accuracy, 0.79 F1-score
- **Features**: TF-IDF, speaker role, policy area, temporal features

### 4. Statistical Modeling
- **Framework**: Generalized Linear Mixed-Effects Models (GLMM)
- **Dependent Variable**: Prominence (binary, from classifier)
- **Levels**: Policy area (Level 3), Interest group (Level 2), Mention (Level 1)
- **Software**: R with `lme4`, `broom.mixed`, `ggplot2`

---

## 📖 Key Concepts

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

### Why Study Prominence?

Traditional lobbying research focuses on **policy outcomes** (did the group win?). But prominence captures:

- **Recognition and credibility** before policy decisions are made
- **Soft power** dynamics in legislative discourse
- **Differential treatment** of groups by policymakers
- **Reputation building** as a resource for future influence

---

## 📚 Documentation

### Essential Reading

1. **[docs/reproducibility_notes.md](./docs/reproducibility_notes.md)** - Step-by-step reproduction guide
2. **[docs/data_dictionary.md](./docs/data_dictionary.md)** - Complete variable reference
3. **[MIGRATION_PLAN.md](./MIGRATION_PLAN.md)** - Understanding the legacy→modern transition
4. **[legacy/README.md](./legacy/README.md)** - Why original code is not reproducible

### API Setup

- **[docs/api_setup_guide.md](./docs/api_setup_guide.md)** - GovInfo and Congress.gov API credentials

### Technical Reports

- **[legacy/5. Visualization and Reporting/Technical Report MA Thesis.pdf](./legacy/5.%20Visualization%20and%20Reporting/Technical%20Report%20MA%20Thesis.pdf)** - Original methodology document
- **[legacy/5. Visualization and Reporting/Thesis_UvA_Kaleb_Mazurek.pdf](./legacy/5.%20Visualization%20and%20Reporting/Thesis_UvA_Kaleb_Mazurek.pdf)** - Full thesis manuscript

---

## 🛠️ Technical Stack

### Current (Legacy)
- **Python 3.7+** (pandas, scikit-learn, BeautifulSoup, requests)
- **R 4.0+** (lme4, ggplot2, broom.mixed)
- Ad-hoc scripts and notebooks

### Planned (Modern Pipeline)
- **Python 3.10+** with pinned dependencies
- **Docker** for environment consistency
- **DVC** for data version control
- **pytest** for automated testing
- **Sphinx** for documentation generation
- **GitHub Actions** for CI/CD

---

## 🤝 Contributing

This project is transitioning to a fully reproducible research pipeline. Contributions are welcome!

### Priority Areas

1. **Testing**: Unit tests for data processing functions
2. **Documentation**: Improving data dictionaries and code comments
3. **Validation**: Cross-checking legacy outputs with new pipeline
4. **Extension**: Collecting data for 2024-2025 Congress

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests and documentation
4. Submit a pull request

See **[CONTRIBUTING.md](./CONTRIBUTING.md)** (coming soon) for detailed guidelines.

---

## 📜 Citation

If you use this code or data in your research, please cite:

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

### Dataset Citation (Multi-Level Data)

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
Email: kalebmazurek@gmail.com
GitHub: [@kmazurek95](https://github.com/kmazurek95)

For questions about:
- **Original thesis methodology**: See [legacy/README.md](./legacy/README.md)
- **Reproducibility**: See [docs/reproducibility_notes.md](./docs/reproducibility_notes.md)
- **Data access**: See [data/README.md](./data/README.md)
- **Technical issues**: [Open a GitHub issue](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Data

- **Congressional Record**: Public domain (U.S. Government work)
- **Washington Representatives Study**: Used with permission
- **Google Trends**: Accessed via public API

---

## 🙏 Acknowledgments

### Institutions
- **University of Amsterdam** - Political Science Department
- **U.S. Government Publishing Office** - GovInfo API access
- **Library of Congress** - Congress.gov API

### Inspiration

This project demonstrates that **research transparency** means acknowledging both successes and limitations:

> "Science is a process, not a product. Reproducibility is something we build toward, not something we achieve on the first try."

### Future Researchers

If you're building on this work:
- Start with the **`multi_level_data`** dataset
- Learn from the **legacy code's limitations**
- Contribute to the **modern pipeline**
- Ask questions via **GitHub issues**

---

## 📊 Project Statistics

- **Original Data Collection**: 77,000+ Congressional Record documents
- **Time Period**: 2014-2018 (114th & 115th Congress)
- **Interest Group Mentions**: ~25,000 extracted mentions
- **Hand-Labeled Training Data**: 2,000 mentions
- **Classification Accuracy**: 81% (SVM model)
- **Statistical Models**: 3 levels (policy/group/mention)

---

**Repository Status**: Modernization in progress (November 2024)
**Last Updated**: November 25, 2024
**Version**: 2.0.0-alpha

---


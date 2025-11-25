# Notebooks Directory

**Purpose**: Exploratory analysis and interactive documentation

**Status**: Selectively version-controlled

---

## Overview

Jupyter notebooks for:
- **Exploratory data analysis** (EDA)
- **Prototyping** new analyses
- **Interactive tutorials** for using the pipeline
- **Validation** of pipeline outputs

---

## Organization

```
notebooks/
├── README.md                              # This file
├── exploration/                           # EDA notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── validation/                            # Validation notebooks
│   ├── compare_speaker_assignments.ipynb
│   ├── compare_mention_extraction.ipynb
│   └── compare_classifier_outputs.ipynb
├── tutorials/                             # How-to guides
│   ├── getting_started.ipynb
│   └── using_multi_level_data.ipynb
└── archive/                               # Old/deprecated notebooks
```

---

## Best Practices

### Naming Convention

```
{number}_{descriptive_name}.ipynb

Examples:
- 01_initial_data_exploration.ipynb
- 02_prominence_patterns_by_policy.ipynb
- 03_lobbying_expenditure_analysis.ipynb
```

### Notebook Structure

```markdown
# Title

**Purpose**: Brief description
**Date**: YYYY-MM-DD
**Author**: Name

## Contents
1. Data Loading
2. Data Cleaning
3. Analysis
4. Conclusions

## Setup

[Import cells]

## Analysis

[Analysis cells with markdown explanations]

## Conclusions

[Summary of findings]
```

### Code Quality

- ✅ Clear markdown explanations between code cells
- ✅ Meaningful variable names
- ✅ Remove or document "magic numbers"
- ✅ Run "Restart & Run All" before committing

---

## Version Control

### What to Commit

✅ **Commit**:
- Notebooks with cleared outputs (use `nbstripout`)
- Tutorial and documentation notebooks
- Final exploratory notebooks with findings

❌ **Don't Commit**:
- Notebooks with large outputs (images, tables)
- Work-in-progress exploratory notebooks
- Notebooks with sensitive data

### Setup nbstripout

Automatically clear outputs before committing:

```bash
pip install nbstripout
nbstripout --install
```

---

## Usage

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab
```

### Converting to Scripts

```bash
# Convert notebook to Python script
jupyter nbconvert --to script notebooks/exploration/01_data_exploration.ipynb

# Convert to HTML for sharing
jupyter nbconvert --to html notebooks/tutorials/getting_started.ipynb
```

---

**Last Updated**: November 25, 2024

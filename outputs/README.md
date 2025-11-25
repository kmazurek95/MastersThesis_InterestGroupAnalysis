# Outputs Directory

**Purpose**: Store analysis outputs, visualizations, and model artifacts

**Status**: Not version-controlled (generated from code)

---

## Directory Structure

```
outputs/
├── README.md                      # This file
├── figures/                       # Publication-ready visualizations
├── tables/                        # Statistical tables and summaries
└── models/                        # Trained model artifacts
```

---

## Folder Details

### `figures/` - Visualizations

**Contents**:
- Plots for manuscripts and presentations
- Exploratory data visualizations
- Model performance charts

**Naming Convention**:
```
{figure_number}_{descriptive_name}_{YYYYMMDD}.{ext}

Examples:
- fig1_prominence_by_policy_area_20241120.png
- fig2_lobbying_expenditure_scatter_20241120.pdf
- fig_supplement_classifier_roc_curve_20241120.svg
```

**Formats**:
- **PNG**: For embedding in documents (300 dpi minimum)
- **PDF**: For publication (vector graphics)
- **SVG**: For web display

---

### `tables/` - Statistical Tables

**Contents**:
- Regression model outputs
- Descriptive statistics
- Summary tables

**Naming Convention**:
```
table{number}_{descriptive_name}_{YYYYMMDD}.{ext}

Examples:
- table1_descriptive_statistics_20241120.csv
- table2_glmm_results_20241120.tex
- table3_classifier_performance_20241120.html
```

**Formats**:
- **CSV**: For data tables
- **LaTeX (.tex)**: For publication-ready tables
- **HTML**: For interactive tables

---

### `models/` - Model Artifacts

**Contents**:
- Trained classifiers
- Feature transformers
- Model metadata

**Naming Convention**:
```
{model_type}_v{version}_{YYYYMMDD}.{ext}

Examples:
- svm_classifier_v1.0_20241120.pkl
- tfidf_vectorizer_v1.0_20241120.pkl
- random_forest_v2.1_20241125.joblib
```

**Formats**:
- **Pickle (.pkl)**: For scikit-learn models
- **Joblib (.joblib)**: For large models (more efficient)
- **JSON**: For model metadata and hyperparameters

---

## Usage

### Generating Outputs

Run analysis scripts to generate outputs:

```bash
# Generate all outputs
python pipeline/04_analysis/generate_visualizations.py
Rscript pipeline/04_analysis/run_multilevel_models.R

# Outputs are saved to respective folders
ls outputs/figures/
ls outputs/tables/
```

### Accessing Outputs

Outputs are not version-controlled (too large, frequently regenerated). To reproduce:

```bash
# Run the full pipeline
python pipeline/run_full_pipeline.py --congress 118 --year 2024

# Or run specific analysis stage
python pipeline/04_analysis/generate_visualizations.py \
    --input data/processed/multi_level_data/multi_level_data_v1.0.parquet \
    --output outputs/figures/
```

---

## Version Control

### What Gets Committed

❌ **Do NOT commit**:
- Generated figures (can be reproduced)
- Large model files (use DVC or external storage)
- Temporary/draft outputs

✅ **Do commit**:
- Code that generates outputs
- Documentation of outputs
- Small reference figures (< 1 MB) if needed for README

### .gitignore Rules

```gitignore
# Ignore all outputs
outputs/**

# Except READMEs
!outputs/README.md
!outputs/*/README.md

# Optionally include small reference figures
!outputs/figures/reference_*.png
```

---

## Publication Checklist

Before submitting outputs for publication:

- [ ] Figures are high resolution (300+ dpi for raster)
- [ ] Font sizes are readable (minimum 8pt)
- [ ] Color schemes are colorblind-friendly
- [ ] Axes are properly labeled with units
- [ ] Captions are descriptive
- [ ] File names match manuscript figure numbers
- [ ] All outputs can be reproduced from code

---

**Last Updated**: November 25, 2024

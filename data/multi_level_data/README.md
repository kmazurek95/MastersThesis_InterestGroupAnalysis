# Multi-Level Data: Reproducibility Anchor Dataset

**Version**: 1.0
**Created**: 2024
**Format**: Parquet (compressed columnar format)
**Size**: TBD

---

## Overview

The **`multi_level_data`** dataset is the **ground truth, validated dataset** that serves as the reproducibility anchor for this project. It contains all processed data from the original thesis analysis (2014-2018 Congressional Record), enabling researchers to reproduce findings without re-running the legacy pipeline.

---

## What's Included

### Hierarchical Structure

The dataset has a hierarchical structure representing:

```
Policy Areas (Level 3)
    └── Interest Groups (Level 2)
        └── Mentions (Level 1)
            └── Paragraphs
                └── Granules (Documents)
```

### Key Variables

#### Document-Level
- `granule_id`: Unique identifier for Congressional Record document
- `congress`: Congress session (114 or 115)
- `date`: Date of debate
- `chamber`: House or Senate
- `session`: Congressional session

#### Paragraph-Level
- `paragraph_id`: Unique identifier for paragraph
- `paragraph_text`: Full text of paragraph
- `speaker_id`: Bioguide ID of speaker
- `speaker_name`: Name of speaker
- `speaker_party`: Party affiliation (D, R, I)
- `speaker_state`: State represented
- `speaker_seniority`: Years in Congress

#### Mention-Level
- `mention_id`: Unique identifier for interest group mention
- `mention_text`: Text of the mention
- `mention_context`: Surrounding text (±50 words)
- `interest_group_name`: Standardized group name
- `interest_group_acronym`: Acronym (if applicable)
- `prominent`: Binary label (1 = prominent, 0 = routine)
- `predicted_prominent`: Classifier prediction
- `prediction_confidence`: Confidence score (0-1)

#### Group-Level
- `group_type`: Type of interest group
- `lobbying_expenditure`: Total lobbying spending
- `membership_size`: Number of members/organizations
- `policy_breadth`: Number of policy areas engaged
- `resource_score`: Composite resource measure

#### Policy-Level
- `policy_area`: Primary policy area
- `policy_salience`: Google Trends salience score
- `salience_category`: Low/Medium/High salience

---

## Data Dictionary

For complete variable definitions, see **[docs/data_dictionary.md](../../../docs/data_dictionary.md)** (Coming in Phase 2).

---

## Loading the Dataset

### Python (pandas)

```python
import pandas as pd

# Load full dataset
data = pd.read_parquet('data/processed/multi_level_data/multi_level_data_v1.0.parquet')

# Explore
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")

# Filter to prominent mentions only
prominent = data[data['prominent'] == 1]
```

### Python (dask for large files)

```python
import dask.dataframe as dd

# Load with dask for out-of-memory processing
data = dd.read_parquet('data/processed/multi_level_data/multi_level_data_v1.0.parquet')

# Process in chunks
result = data.groupby('policy_area')['prominent'].mean().compute()
```

### R

```r
library(arrow)

# Load parquet file
data <- read_parquet("data/processed/multi_level_data/multi_level_data_v1.0.parquet")

# Explore
dim(data)
str(data)
summary(data$prominent)

# Filter prominent mentions
prominent <- data %>% filter(prominent == 1)
```

---

## Data Quality

### Known Limitations

1. **Speaker Assignment Accuracy**: Estimated ~85-90% (not validated against ground truth)
2. **Mention Extraction Recall**: Unknown (no complete ground truth for all groups)
3. **Classifier Performance**: 81% accuracy, 0.79 F1-score on test set
4. **Missing Values**: Some metadata fields incomplete (see data dictionary)

### Quality Checks Performed

✅ No duplicate `mention_id` values
✅ All `granule_id` values present in document table
✅ Speaker IDs validated against Congress.gov
✅ Dates within expected range (2014-2018)
✅ Policy areas mapped consistently

### Validation Report

See `validation_report_v1.0.html` (when generated in Phase 2) for:
- Missing data patterns
- Outlier detection
- Distribution summaries
- Cross-validation results

---

## Example Analyses

### 1. Prominence by Policy Area

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate prominence rate by policy area
prominence_by_area = (
    data.groupby('policy_area')['prominent']
    .agg(['mean', 'count'])
    .sort_values('mean', ascending=False)
)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=prominence_by_area.index, y=prominence_by_area['mean'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Prominence Rate')
plt.title('Interest Group Prominence by Policy Area')
plt.tight_layout()
plt.show()
```

### 2. Lobbying Expenditure vs. Prominence

```python
# Aggregate to group level
group_data = (
    data.groupby('interest_group_name')
    .agg({
        'prominent': 'mean',
        'lobbying_expenditure': 'first',
        'membership_size': 'first'
    })
)

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(
    group_data['lobbying_expenditure'],
    group_data['prominent'],
    alpha=0.5
)
plt.xlabel('Lobbying Expenditure ($)')
plt.ylabel('Prominence Rate')
plt.xscale('log')
plt.title('Lobbying Expenditure vs. Prominence')
plt.show()
```

### 3. Multilevel Model in R

```r
library(lme4)
library(broom.mixed)

# Fit GLMM
model <- glmer(
    prominent ~
        lobbying_expenditure + policy_breadth + membership_size +
        speaker_seniority + speaker_party +
        policy_salience +
        (1 | policy_area) + (1 | interest_group_name),
    data = data,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa")
)

# Summarize
summary(model)
tidy(model, effects = "fixed")
```

---

## File Location

### Current Status

⚠️ **The actual data file may not be in the Git repository due to size.**

### How to Access

**Option A: Request from Author**
- Email kalebmazurek@gmail.com
- Specify your use case and institutional affiliation
- Large file may be shared via institutional server or cloud storage

**Option B: Generate from Pipeline** (Phase 3+)
- Once the modern pipeline is complete, you can regenerate:
```bash
python pipeline/run_full_pipeline.py --congress 114-115 --years 2014-2018
```

**Option C: Use DVC** (Planned)
- After DVC implementation, pull the data:
```bash
dvc pull data/processed/multi_level_data.dvc
```

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{mazurek2024multilevel,
  author  = {Mazurek, Kaleb},
  title   = {Multi-Level Interest Group Prominence Dataset: U.S. Congressional Record 2014-2018},
  year    = {2024},
  version = {1.0},
  url     = {https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis}
}
```

And the original thesis:

```bibtex
@mastersthesis{mazurek2023prominence,
  author  = {Mazurek, Kaleb},
  title   = {Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence},
  school  = {University of Amsterdam},
  year    = {2023},
  type    = {Master's Thesis}
}
```

---

## Version History

### v1.0 (2024)
- Initial release
- Covers 114th and 115th Congress (2014-2018)
- Includes all processed variables from original thesis
- Validated against thesis results

---

## Contact

For questions about this dataset:

**Kaleb Mazurek**
Email: kalebmazurek@gmail.com
GitHub: [@kmazurek95](https://github.com/kmazurek95)

---

**Last Updated**: November 25, 2024

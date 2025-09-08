# Master's Thesis: Interest Group Analysis Repository

This repository contains a complete academic data analysis pipeline for the Master's thesis "Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence" (University of Amsterdam, 2023). The codebase analyzes ~78,000 legislative documents using machine learning and statistical modeling to understand interest group prominence in U.S. Congress.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the information here.**

## Working Effectively

### Bootstrap and Setup Repository:
1. **Install Python dependencies**:
   ```bash
   pip3 install requests beautifulsoup4 tqdm retrying pandas numpy seaborn matplotlib scikit-learn statsmodels datasketch nltk joblib
   ```
   Takes 2-3 minutes. NEVER CANCEL.

2. **Download required NLTK data**:
   ```bash
   python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
   ```
   Takes 30 seconds.

3. **Install R (for statistical modeling)**:
   ```bash
   sudo apt-get update && sudo apt-get install -y r-base
   ```
   Takes 5-8 minutes. NEVER CANCEL. Set timeout to 600+ seconds.

4. **Install R packages** (when network permits):
   ```bash
   echo 'install.packages(c("lme4", "dplyr", "ggplot2", "broom.mixed", "kableExtra", "forcats", "assertthat", "knitr", "tidyr"), repos="https://cran.rstudio.com/")' | R --slave
   ```
   R package installation may fail due to network restrictions in sandboxed environments.

### Key Environment Limitations:
- **CRITICAL**: API access to GovInfo, Congress API, and Google Trends is blocked in sandboxed environments due to network restrictions.
- Data collection scripts require API keys and external network access.
- R CRAN package installation may fail due to network restrictions.
- Scripts are designed for offline execution with pre-collected data.

## Repository Structure and Workflow

Navigate to subfolders sequentially as this is a **5-stage research pipeline**:

```bash
MasterThesisUniversityOfAmsterdam/
├── 1. Data Collection/          # API-based data gathering (requires network access)
├── 2. Data Processing/          # Data cleaning and preparation 
├── 3. Supervised Learning Classifiers/  # ML text classification pipeline
├── 4. Integrated Dataset and Analysis/  # Statistical modeling (Python + R)
└── 5. Visualization and Reporting/      # Final reports and thesis documents
```

## Working with Each Stage

### 1. Data Collection (Network Dependent)
**Location**: `MasterThesisUniversityOfAmsterdam/1. Data Collection/`

**Scripts**:
- `govinfo_data_fetcher.py` - Fetches legislative transcripts from GovInfo API
- `fetch_bill_data.py` - Collects bill metadata
- `congress_member_data_collector.py` - Retrieves congressional member profiles  
- `policy_salience_pipeline.py` - Analyzes policy salience via Google Trends

**IMPORTANT**: These scripts require API keys and external network access. In sandboxed environments, they will fail with DNS resolution errors.

**Expected Runtime**: 
- Each script: 30 minutes to 4+ hours depending on data volume. NEVER CANCEL.
- Set timeouts to 240+ minutes for data collection operations.

**Usage**:
```bash
cd "MasterThesisUniversityOfAmsterdam/1. Data Collection/"
python3 govinfo_data_fetcher.py  # Will fail without network access
```

### 2. Data Processing  
**Location**: `MasterThesisUniversityOfAmsterdam/2. Data Processing/`

**Key Scripts**:
- `process_api_results.py` - Processes API responses
- `assign_speakers_to_granules.py` - Links legislative text to speakers
- `extract_interest_group_mentions.py` - Identifies organization mentions
- `policy_area_mapper.py` - Maps committees to policy areas

**Expected Runtime**: Each script 5-45 minutes. NEVER CANCEL. Set timeout to 60+ minutes.

**Usage**:
```bash
cd "MasterThesisUniversityOfAmsterdam/2. Data Processing/"
python3 process_api_results.py
```

### 3. Supervised Learning Classifiers
**Location**: `MasterThesisUniversityOfAmsterdam/3. Supervised Learning Classifiers/`

**Main Script**: `text_classifier_pipeline.py`

**Features**:
- Text preprocessing with NLTK
- Multiple ML models: Naive Bayes, Logistic Regression, SVM, Random Forest
- Grid search hyperparameter optimization
- Achieves 81% accuracy with SVM classifier

**Expected Runtime**: 15-60 minutes for full pipeline. NEVER CANCEL. Set timeout to 90+ minutes.

**Prerequisites**:
- NLTK data must be downloaded (see bootstrap steps)
- Requires CSV files with columns: `p1_original` (text), `prominence` (labels)

**Usage**:
```bash
cd "MasterThesisUniversityOfAmsterdam/3. Supervised Learning Classifiers/"
python3 text_classifier_pipeline.py
```

### 4. Integrated Dataset and Analysis
**Location**: `MasterThesisUniversityOfAmsterdam/4. Integrated Dataset and Analysis/`

**Python Scripts**:
- `DataProcessingAndRegression.py` - Data preprocessing and regression analysis
- `InterestGroupAnalysisPipeline.py` - Data integration and validation

**R Scripts**:
- `model_policy_salience.R` - Models salience-related factors
- `model_group_politician.R` - Models politician-group linkages  
- `model_group_characteristics.R` - Models group-level attributes

**Expected Runtime**: 
- Python scripts: 10-30 minutes each. NEVER CANCEL. Set timeout to 45+ minutes.
- R scripts: 5-20 minutes each. NEVER CANCEL. Set timeout to 30+ minutes.

**Usage**:
```bash
cd "MasterThesisUniversityOfAmsterdam/4. Integrated Dataset and Analysis/"
python3 DataProcessingAndRegression.py
R --slave < model_policy_salience.R
```

### 5. Visualization and Reporting
**Location**: `MasterThesisUniversityOfAmsterdam/5. Visualization and Reporting/`

Contains final thesis documents:
- `Technical Report MA Thesis.pdf` - Technical summary
- `Thesis_UvA_Kaleb_Mazurek.pdf` - Complete thesis document

## Validation Scenarios

After making changes, ALWAYS test these scenarios:

### Python Environment Validation:
```bash
python3 -c "
import pandas as pd
import numpy as np  
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
print('All core dependencies work')
"
```

### Text Processing Validation:
```bash
python3 -c "
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower().strip())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

test = 'The American Medical Association supports healthcare legislation.'
result = preprocess_text(test)
print(f'Text processing works: \"{result}\"')
"
```

### R Basic Functionality:
```bash
echo 'print("R is working"); data.frame(x=1:3, y=4:6)' | R --slave
```

## Common Tasks Reference

### Check Repository Structure:
```bash
ls -la MasterThesisUniversityOfAmsterdam/
```
Expected output:
```
1. Data Collection/
2. Data Processing/  
3. Supervised Learning Classifiers/
4. Integrated Dataset and Analysis/
5. Visualization and Reporting/
```

### Verify Python Environment:
```bash
python3 --version  # Should be 3.6+
pip3 list | grep -E "(pandas|numpy|scikit-learn|nltk)"
```

### Check NLTK Data:
```bash
python3 -c "import nltk; print(nltk.data.path)"
ls ~/nltk_data/corpora/  # Should contain stopwords
```

### Test ML Pipeline Performance:
Small dataset (1000 samples): ~0.1 seconds total
Medium dataset (10,000 samples): ~2-5 seconds  
Large dataset (100,000+ samples): ~2-10 minutes

## Critical Timing Expectations

**NEVER CANCEL any of these operations:**

- **Environment setup**: 10-15 minutes total
- **Data collection**: 2-8 hours (network dependent, will fail in sandbox)
- **Data processing**: 30 minutes - 2 hours  
- **ML classification**: 15-90 minutes
- **Statistical modeling**: 15-45 minutes
- **Full pipeline**: 4-12 hours (with real data and network access)

**Always set appropriate timeouts:**
- Setup commands: 600+ seconds
- Processing scripts: 3600+ seconds (1 hour)
- Full data collection: 14400+ seconds (4 hours)

## Important Notes for Agents

1. **This is a research codebase, not production software** - no traditional build/test systems
2. **Sequential execution required** - stages depend on outputs from previous stages
3. **Network restrictions affect data collection** - API scripts will fail in sandboxed environments
4. **Large datasets expected** - processing times scale with data volume
5. **Academic focus** - outputs are research datasets and statistical models, not applications
6. **API keys needed** - GovInfo and Congress API keys required for data collection
7. **R package installation may fail** - due to network restrictions in sandboxed environments

## Key File Locations

- **Main README**: `/README.md`
- **Data Collection Scripts**: `/MasterThesisUniversityOfAmsterdam/1. Data Collection/`
- **ML Pipeline**: `/MasterThesisUniversityOfAmsterdam/3. Supervised Learning Classifiers/text_classifier_pipeline.py`
- **Statistical Models**: `/MasterThesisUniversityOfAmsterdam/4. Integrated Dataset and Analysis/*.R`
- **Final Thesis**: `/MasterThesisUniversityOfAmsterdam/5. Visualization and Reporting/Thesis_UvA_Kaleb_Mazurek.pdf`

**Contact**: For questions about this codebase, contact Kaleb Mazurek at kalebmazurek@gmail.com
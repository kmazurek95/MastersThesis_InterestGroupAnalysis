# Legacy Code Archive

**⚠️ IMPORTANT NOTICE ⚠️**

This folder contains the **original codebase** from the master's thesis project completed in 2023. The code is preserved **exactly as it existed** during the original research for historical reference and transparency.

---

## Purpose of This Archive

This legacy code represents the exploratory, iterative research process that produced the findings in:

> Mazurek, Kaleb. *Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence*. University of Amsterdam, 2023.

**This code is NOT fully reproducible** and is archived here to:

1. **Preserve Research History**: Document the original methodology and decisions
2. **Provide Context**: Show the evolution from exploratory analysis to reproducible pipeline
3. **Maintain Transparency**: Acknowledge that thesis research often involves manual steps
4. **Support Future Researchers**: Offer insights into the original workflow decisions

---

## Why This Code Is Not Reproducible

### 1. **Manual Data Processing Steps**
- Hand-labeled training data for supervised learning classifiers
- Manual verification and correction of interest group mentions
- Ad-hoc data cleaning decisions not captured in code
- Manual merging of external datasets (Washington Representatives Study, Google Trends)

### 2. **Missing Intermediate Data**
- The GovInfo API raw outputs are not included (too large for GitHub)
- Intermediate processing files were created and modified manually
- Some data transformations occurred in interactive Python sessions
- External datasets obtained from third-party sources are not version-controlled

### 3. **Environment and Dependency Issues**
- No comprehensive `requirements.txt` with pinned versions
- Scripts developed in Jupyter notebooks then converted to `.py` files
- Assumes specific local file paths and directory structures
- API keys and authentication details not standardized

### 4. **Non-Linear Workflow**
- Scripts were not always run in sequential order
- Some scripts were re-run with different parameters
- Data exploration led to retroactive changes in earlier steps
- File naming conventions evolved during the project

### 5. **Undocumented Assumptions**
- Threshold values for classification were determined through experimentation
- Text preprocessing decisions (stopwords, lemmatization) evolved iteratively
- Speaker assignment logic included manual edge-case handling
- Policy area mappings required subjective categorization

---

## Original Folder Structure

```
legacy/
├── 1. Data Collection/
│   ├── govinfo_data_fetcher.py          # Fetched Congressional Record transcripts
│   ├── fetch_bill_data.py               # Collected bill metadata
│   ├── congress_member_data_collector.py # Gathered member profiles
│   └── policy_salience_pipeline.py      # Google Trends analysis
│
├── 2. Data Proccessing/                 # [Note: Original typo preserved]
│   ├── process_api_results.py           # Initial API data processing
│   ├── crec_data_expander.py            # Expanded Congressional Record data
│   ├── assign_speakers_to_paragraphs.py # Speaker-text matching
│   ├── extract_interest_group_mentions.py # IG mention extraction
│   ├── tfidf_duplicate_mention_finder.py # Duplicate detection
│   └── [14 additional processing scripts]
│
├── 3. Supervised Learning Classifiers/
│   └── text_classifier_pipeline.py      # SVM/Naive Bayes/Random Forest models
│
├── 4. Integrated Dataset and Analysis/
│   ├── InterestGroupAnalysisPipeline.py # Main analysis orchestration
│   └── DataProcessingAndRegression.py   # Statistical modeling prep
│
└── 5. Visualization and Reporting/
    ├── Thesis_UvA_Kaleb_Mazurek.pdf     # Final thesis document
    └── Technical Report MA Thesis.pdf    # Technical methodology report
```

---

## What the Original Pipeline Did

### Stage 1: Data Collection (2014-2018 Congressional Record)
- Fetched ~77,000 legislative documents via GovInfo API
- Collected congressional member profiles
- Gathered bill metadata and policy area classifications
- Obtained Google Trends data for policy salience measures

### Stage 2: Data Processing
- Parsed HTML/XML transcripts into structured text
- Assigned speakers to paragraphs using heuristic rules
- Extracted interest group mentions using pattern matching and NER
- Removed duplicate mentions using TF-IDF similarity
- Mapped bills and debates to policy areas

### Stage 3: Supervised Learning
- Hand-labeled ~2,000 mentions as "prominent" vs "non-prominent"
- Trained text classifiers (SVM achieved ~81% accuracy)
- Generated predictions for all unlabeled mentions
- Created labeled prominence dataset

### Stage 4: Statistical Modeling
- Integrated classified mentions with speaker metadata
- Added group-level variables (lobbying expenditure, membership)
- Added policy-level variables (salience, topic)
- Ran Generalized Linear Mixed-Effects Models in R

### Stage 5: Analysis & Reporting
- Produced thesis manuscript and technical report
- Generated visualizations and summary statistics

---

## Known Limitations and Issues

### Data Quality
- **Speaker assignment accuracy**: ~85-90% (estimated, not validated)
- **Interest group mention recall**: Unknown (no ground truth dataset)
- **Duplicate detection**: Conservative approach may have missed true duplicates
- **Policy area mapping**: Manual categorization with subjective boundaries

### Reproducibility Gaps
- **No single entry point**: No `main.py` to run the entire pipeline
- **Hard-coded paths**: Scripts assume specific local directory structures
- **Inconsistent data formats**: Some outputs in CSV, others in JSON or pickle
- **Missing documentation**: Parameter choices and thresholds undocumented
- **No versioning**: Data and code not synchronized with version control

### Technical Debt
- **Code duplication**: Similar logic repeated across multiple scripts
- **No unit tests**: Functions not validated with automated tests
- **Error handling**: Minimal exception handling and validation
- **Memory inefficiency**: Some scripts load entire datasets into RAM
- **No logging**: Limited debugging information in outputs

---

## How the New Pipeline Addresses These Issues

The modern `/pipeline/` directory addresses legacy limitations through:

1. **Reproducibility Anchor**: Uses `multi_level_data` as the validated ground truth
2. **Modular Design**: Clear separation of concerns with testable functions
3. **Documentation**: Comprehensive docstrings and data dictionaries
4. **Version Control**: All code and data versions tracked
5. **Environment Management**: Containerization and dependency pinning
6. **Validation**: Unit tests and data quality checks
7. **Transparency**: Explicit documentation of assumptions and decisions

---

## Using Legacy Code

### ⚠️ Not Recommended for Reproduction

These scripts **should not be used** to reproduce the thesis results. Instead:

- Use the **`/data/processed/multi_level_data/`** dataset as the starting point
- Reference the **`/pipeline/`** directory for modern, reproducible code
- Consult **`/docs/reproducibility_notes.md`** for detailed guidance

### When to Reference Legacy Code

✅ **Do use legacy code to**:
- Understand the original research decisions
- Learn about the data collection methodology
- See the evolution of the analysis approach
- Debug questions about the original thesis findings

❌ **Do not use legacy code to**:
- Reproduce the thesis results
- Collect new data for recent Congress sessions
- Build production pipelines
- Teach reproducible research practices

---

## Thesis Citation

If referencing the original thesis methodology or findings, please cite:

```bibtex
@mastersthesis{mazurek2023prominence,
  author  = {Mazurek, Kaleb},
  title   = {Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence},
  school  = {University of Amsterdam},
  year    = {2023},
  type    = {Master's Thesis},
  note    = {Available at: https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis}
}
```

---

## Questions About Legacy Code?

For questions about the original methodology or code:

- **Email**: kalebmazurek@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues) with the `legacy-code` label

---

## Acknowledgments

This legacy archive demonstrates the reality of academic research:

> "Real research is messy, iterative, and exploratory. Reproducibility is a goal we strive toward, not a starting point."

Thank you to the University of Amsterdam for supporting this work, and to future researchers who will build upon these foundations with improved practices.

---

**Last Updated**: November 2024
**Archive Created**: Part of 2024 repository modernization initiative

# Legacy Code Archive

This folder contains the original codebase from the master's thesis completed in 2023. The code is preserved as it existed during the research for historical reference and transparency.

## Why This Code Is Not Reproducible

1. **Manual data processing steps**: Hand-labeled training data, manual verification of mentions, ad-hoc cleaning decisions not captured in code
2. **Missing intermediate data**: GovInfo API raw outputs not included, intermediate files created interactively
3. **Environment issues**: No pinned dependency versions, scripts assume specific local paths
4. **Non-linear workflow**: Scripts were not always run sequentially, parameters changed through experimentation
5. **Undocumented assumptions**: Classification thresholds, text preprocessing decisions, and policy area mappings evolved iteratively

## Folder Structure

```
legacy/
├── 1. Data Collection/
│   ├── govinfo_data_fetcher.py          # Fetched Congressional Record transcripts
│   ├── fetch_bill_data.py               # Collected bill metadata
│   ├── congress_member_data_collector.py # Gathered member profiles
│   └── policy_salience_pipeline.py      # Google Trends analysis
│
├── 2. Data Processing/
│   ├── process_api_results.py           # Initial API data processing
│   ├── crec_data_expander.py            # Expanded Congressional Record data
│   ├── assign_speakers_to_paragraphs.py # Speaker-text matching
│   ├── extract_interest_group_mentions.py # IG mention extraction
│   ├── tfidf_duplicate_mention_finder.py # Duplicate detection
│   └── [additional processing scripts]
│
├── 3. Supervised Learning Classifiers/
│   └── text_classifier_pipeline.py      # SVM/Naive Bayes/Random Forest models
│
├── 4. Integrated Dataset and Analysis/
│   ├── InterestGroupAnalysisPipeline.py # Main analysis orchestration
│   ├── DataProcessingAndRegression.py   # Statistical modeling prep
│   ├── model_group_characteristics.R    # GLMM: organizational characteristics
│   ├── model_group_politician.R         # GLMM: politician-group linkage
│   └── model_policy_salience.R          # GLMM: policy salience effects
│
└── 5. Visualization and Reporting/
    ├── Thesis_UvA_Kaleb_Mazurek.pdf     # Final thesis document
    └── Technical Report MA Thesis.pdf    # Technical methodology report
```

## What the Pipeline Did

1. **Data Collection**: Fetched ~77,000 legislative documents via GovInfo API, collected member profiles and bill metadata, obtained Google Trends data for policy salience
2. **Data Processing**: Parsed transcripts, assigned speakers to paragraphs, extracted interest group mentions, removed duplicates, mapped policy areas
3. **Supervised Learning**: Hand-labeled ~2,000 mentions, trained classifiers (SVM achieved ~81% accuracy), generated predictions for all mentions
4. **Statistical Modeling**: Integrated data with speaker and group metadata, ran GLMMs in R
5. **Reporting**: Produced thesis manuscript and technical report

## Using This Code

These scripts should not be used to reproduce the thesis results. Use the analysis notebooks in `/analysis/` with the validated dataset in `/data/multi_level_data/` instead.

This code is useful for understanding the original research decisions, learning about the data collection methodology, and seeing how the analysis evolved.

Note: API keys have been replaced with placeholders. Hardcoded file paths have been replaced with environment variable lookups.

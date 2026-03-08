# Analysis Dataset

This folder contains the processed dataset used in the thesis analysis. Files are tracked with Git LFS due to size.

## Files

- `level1_FINAL.csv` (~107 MB): Primary analysis dataset. 19,165 interest group mentions with 175 variables covering organization characteristics, politician attributes, policy area metadata, and prominence classifications.
- `df_interest_group_prominence_FINAL.csv`: Prominence-specific subset used in some analysis steps.
- `multi_level_data.csv`: Alternate format of the same data.

## How to access

After cloning, run `git lfs pull` to download the actual CSV files. Without this step, you'll see pointer files instead of data.

## Unit of analysis

Each row is one mention of an interest group in a congressional floor speech paragraph. The dataset covers the 114th and 115th Congress (2015-2019).

## Key variables

- `level1_prominence`: Binary (1 = prominent citation, 0 = routine mention). Classified by SVM at ~81% accuracy.
- `level1_org_id`: Organization identifier from the Washington Representatives Study.
- `saliency_measure` / `saliency_category`: Policy salience from Google Trends data.
- `level1_chamber_x`: House or Senate.
- `level1_partyHistory`: Speaker party affiliation.
- `level1_ABBREVCAT`: Organization type category.

See the analysis notebooks in `../analysis/` for how these variables are used in the models.

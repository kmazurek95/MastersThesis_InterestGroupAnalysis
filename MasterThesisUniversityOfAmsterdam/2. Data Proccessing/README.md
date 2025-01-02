# 2. Data Processing

## Overview
The **Data Processing** folder contains scripts designed to refine, clean, and organize data collected during the **Data Collection** phase. These scripts focus on tasks such as expanding nested data, linking speakers to legislative text, highlighting mentions, and analyzing interest group prominence. This stage prepares the data for advanced analysis and visualization.

---

## Scripts Overview

### 1. `2.process_api_results.py`
**Purpose**: Processes raw API results and prepares them for downstream analysis.
- **Key Features**:
  - Cleans and normalizes API responses.
  - Extracts key fields and metadata.
  - Handles incomplete or missing data.
- **Outputs**:
  - CSV and JSON files with cleaned API data.

---

### 2. `3.process_congressional_granules.py`
**Purpose**: Organizes and processes congressional granule data for analysis.
- **Key Features**:
  - Parses granule text and extracts metadata.
  - Links granules to congressional sessions and topics.
- **Outputs**:
  - CSV files containing structured granule information.

---

### 3. `4.crec_data_expander.py`
**Purpose**: Expands CREC (Congressional Record) data into analyzable formats.
- **Key Features**:
  - Flattens nested CREC fields such as topics and references.
  - Creates structured datasets for further exploration.
- **Outputs**:
  - CSV files with expanded CREC data.

---

### 4. `5.process_crec_data.py`
**Purpose**: Processes CREC data to create a unified dataset.
- **Key Features**:
  - Combines data from multiple CREC sessions.
  - Standardizes text formats and metadata fields.
- **Outputs**:
  - Consolidated CSV files with standardized CREC data.

---

### 5. `6.assign_speakers_to_paragraphs.py`
**Purpose**: Matches speakers to individual paragraphs in legislative text.
- **Key Features**:
  - Analyzes paragraph content to identify speaker associations.
  - Links speaker metadata to text for context.
- **Outputs**:
  - CSV files linking paragraphs to identified speakers.

---

### 6. `8.assign_speakers_to_granules.py`
**Purpose**: Maps speakers to granules based on metadata and context.
- **Key Features**:
  - Identifies primary speakers for each granule.
  - Highlights speaker relevance and roles.
- **Outputs**:
  - CSV files with granule-to-speaker mappings.

---

### 7. `9.tfidf_duplicate_mention_finder.py`
**Purpose**: Detects duplicate mentions in legislative text using TF-IDF and cosine similarity.
- **Key Features**:
  - Highlights paragraphs with overlapping content.
  - Scores similarity between mentions.
- **Outputs**:
  - CSV files detailing duplicate mentions and their similarity scores.

---

### 8. `acronym_highlighter.py`
**Purpose**: Highlights acronyms within legislative text.
- **Key Features**:
  - Identifies and marks acronyms for analysis.
  - Links acronyms to metadata and context.
- **Outputs**:
  - Annotated CSV and JSON files with highlighted acronyms.

---

### 9. `data_labeling_pipeline.py`
**Purpose**: Facilitates the labeling of legislative data for supervised learning models.
- **Key Features**:
  - Organizes data for efficient labeling.
  - Exports labeled datasets for analysis.
- **Outputs**:
  - CSV files ready for machine learning pipelines.

---

### 10. `extract_interest_group_mentions.py`
**Purpose**: Identifies and extracts mentions of interest groups in legislative text.
- **Key Features**:
  - Maps mentions to granules and metadata.
  - Highlights patterns in interest group references.
- **Outputs**:
  - CSV and JSON files detailing interest group mentions.

---

### 11. `granule_data_processor_speaker.py`
**Purpose**: Processes granule data to associate it with speaker metadata.
- **Key Features**:
  - Cleans and structures speaker-granule associations.
  - Prepares data for downstream modeling.
- **Outputs**:
  - Structured CSV files linking speakers and granules.

---

### 12. `granule_speaker_identifier.py`
**Purpose**: Links speakers to granules using textual analysis.
- **Key Features**:
  - Identifies single and multiple speakers in granules.
  - Validates associations with metadata.
- **Outputs**:
  - CSV files summarizing speaker-granule relationships.

---

### 13. `merging_labeled_and_classified_data.py`
**Purpose**: Combines labeled datasets with classified outputs for integration.
- **Key Features**:
  - Merges human-labeled and machine-classified datasets.
  - Resolves conflicts and inconsistencies.
- **Outputs**:
  - Unified CSV files with merged data.

---

### 14. `name_and_acronym_combiner.py`
**Purpose**: Combines data on organization names and acronyms for consistency.
- **Key Features**:
  - Merges datasets across multiple sessions.
  - Normalizes name and acronym usage.
- **Outputs**:
  - Combined CSV and JSON files.

---

### 15. `name_mention_highlighter.py`
**Purpose**: Highlights organization names in legislative text.
- **Key Features**:
  - Identifies and marks organization mentions.
  - Links names to metadata for context.
- **Outputs**:
  - Annotated CSV files with highlighted names.

---

### 16. `nested_data_expander.py`
**Purpose**: Expands nested fields in JSON data into flat tables.
- **Key Features**:
  - Extracts key details from nested structures.
  - Produces analyzable datasets.
- **Outputs**:
  - Flattened CSV files.

---

### 17. `policy_area_mapper.py`
**Purpose**: Maps committees and topics to policy areas.
- **Key Features**:
  - Categorizes legislative data by policy relevance.
  - Summarizes activity by area.
- **Outputs**:
  - CSV files summarizing policy areas.

---

## Workflow
1. **Preprocessing**:
   - Start with `2.process_api_results.py` and `3.process_congressional_granules.py` to clean and organize data.
2. **Speaker Association**:
   - Use `6.assign_speakers_to_paragraphs.py` and `8.assign_speakers_to_granules.py` for speaker-text mappings.
3. **Highlighting and Mapping**:
   - Run `acronym_highlighter.py` and `policy_area_mapper.py` for text analysis.
4. **Mention Extraction**:
   - Execute `extract_interest_group_mentions.py` and `name_mention_highlighter.py` to extract references.

---

## Prerequisites
- **Python**: Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
- **Inputs**: Use outputs from the **Data Collection** stage.
- **Dependencies**: Ensure modules like `pandas`, `numpy`, and `sklearn` are installed.

---

## Contact
For questions or issues, contact **Kaleb Mazurek** at kalebmazurek@gmail.com

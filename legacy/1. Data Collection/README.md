# 1. Data Collection

## Overview
The `Data Collection` folder contains scripts designed to gather and prepare raw legislative, policy, and congressional data. These scripts form the foundational stage of the research pipeline, enabling subsequent analyses and modeling.

## Scripts Overview

### **1. `1.govinfo_data_fetcher.py`**
- **Purpose**: Fetches legislative data from the GovInfo API, including transcripts of debates for the 114th and 115th U.S. Congresses.
- **Key Features**:
  - Implements retry logic for robust API requests.
  - Extracts and parses text content from HTML documents.
  - Processes granule details in parallel for efficiency.
  - Supports resuming data collection using a progress tracker.
- **Outputs**:
  - CSV and JSON files containing legislative transcripts and metadata.
  - Intermediate data for use in downstream analyses.

### **2. `7.fetch_bill_data.py`**
- **Purpose**: Collects metadata about congressional bills using the GovInfo API.
- **Key Features**:
  - Constructs API links dynamically for each bill.
  - Retrieves bill text and metadata.
  - Handles errors and logs failed requests for troubleshooting.
- **Outputs**:
  - CSV file with metadata and links for bills.
  - Detailed JSON responses for further analysis.

### **3. `congress_member_data_collector.py`**
- **Purpose**: Retrieves detailed profiles of congressional members, including party affiliation and legislative activities.
- **Key Features**:
  - Fetches data from the Congress API using bioguide IDs.
  - Normalizes nested JSON responses into flat, analyzable tables.
  - Processes party history and congressional terms for specified sessions.
- **Outputs**:
  - Cleaned CSV file with congress member profiles.
  - JSON file with raw API responses for validation.

### **4. `policy_salience_pipeline.py`**
- **Purpose**: Analyzes policy salience using Google Trends data to gauge public interest in legislative issues.
- **Key Features**:
  - Retrieves search interest data for multiple policy topics.
  - Aggregates trends data by policy area and year.
  - Merges Google Trends data with legislative datasets for analysis.
- **Outputs**:
  - Combined CSV file with Google Trends data.
  - Processed salience metrics by year and congress.
  - Visualizations of trends over time.

## Workflow
1. **Legislative Data Collection**:
   - Run `1.govinfo_data_fetcher.py` to fetch legislative transcripts.
2. **Bill Metadata Collection**:
   - Execute `7.fetch_bill_data.py` to retrieve bill details.
3. **Congress Member Profiles**:
   - Use `congress_member_data_collector.py` to gather member profiles.
4. **Policy Salience Analysis**:
   - Run `policy_salience_pipeline.py` to collect and analyze trends data.

## Prerequisites
- **Python**:
  - Install necessary libraries using `pip install -r requirements.txt`.
  - Key dependencies: `requests`, `pandas`, `tqdm`, `BeautifulSoup4`, `pytrends`.
- **API Keys**:
  - Obtain API keys for GovInfo, Congress API, and Google Trends.
  - Set up environment variables or update script constants with your keys.

## Outputs
- **Legislative Data**:
  - JSON and CSV files with transcripts and granule metadata.
- **Bill Metadata**:
  - CSV file linking bills to policy areas and summaries.
- **Congress Member Profiles**:
  - Detailed CSV files with member affiliations and legislative activities.
- **Policy Salience**:
  - Trends data and salience metrics visualized in CSV files and plots.

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/username/MastersThesis_InterestGroupAnalysis.git
   ```
2. Navigate to the `1. Data Collection` folder:
   ```bash
   cd MasterThesisUniversityOfAmsterdam/1. Data Collection/
   ```
3. Run scripts sequentially for a complete data collection pipeline:
   ```bash
   python 1.govinfo_data_fetcher.py
   python 7.fetch_bill_data.py
   python congress_member_data_collector.py
   python policy_salience_pipeline.py
   ```

## Notes
- Ensure proper API rate-limiting to avoid service interruptions.
- Verify API keys and dependencies before executing scripts.
- Intermediate outputs are saved to assist in debugging and resuming tasks.

## Contact
For questions or issues, contact **Kaleb Mazurek** at kalebmazurek@gmail.com



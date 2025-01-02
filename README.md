# Master’s Thesis: Interest Group Analysis Repository

## Overview
This repository contains the complete workflow for the master's thesis, **"Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence,"** conducted at the University of Amsterdam. The study investigates the dynamics of interest group prominence in legislative debates, leveraging advanced statistical methods, supervised learning, and integrated datasets.

Prominence, a key concept in this study, reflects the recognition and soft power afforded to advocacy organizations by policymakers. By analyzing legislative debates from the 114th and 115th U.S. Congress sessions, this study identifies the factors influencing prominence and challenges conventional notions about interest group success.

---

## Research Questions
1. Why do some politicians prioritize certain advocacy organizations?
2. What factors contribute to variations in prominence across groups and issues?

---

## Skills Demonstrated
This project highlights the following technical and analytical skills:

- **Data Collection**:
  - API usage (e.g., GovInfo API, Congress API, Google Trends).
  - Handling large legislative datasets (~77,000 documents).
- **Data Processing**:
  - Cleaning, transforming, and organizing data for analysis using Python.
- **Machine Learning**:
  - Supervised learning for text classification (Support Vector Machines, Naive Bayes, Random Forests).
- **Statistical Analysis**:
  - Generalized Linear Mixed-Effects Modeling using R.
- **Data Integration**:
  - Combining structured and unstructured data.
- **Visualization and Reporting**:
  - Creating visual insights and documenting findings in structured reports.
- **Project Management**:
  - Multi-stage analysis pipeline management and reproducibility.

---

## Repository Structure

```bash
.
├── LICENSE                                # License details for this repository
├── README.md                              # High-level overview of the repository
├── MasterThesisUniversityOfAmsterdam/     # Main repository directory
│   ├── 1. Data Collection/                # Scripts for collecting legislative, policy, and congress data
│   ├── 2. Data Processing/                # Scripts for cleaning and preparing collected data
│   ├── 3. Supervised Learning Classifiers/ # Machine learning pipeline for text classification
│   ├── 4. Modeling and Analysis/          # Data integration and statistical modeling
│   ├── 5. Visualization and Reporting/    # Reports, visualizations, and thesis documents
│   └── results/                           # Outputs including processed data, models, and plots
```

---

## Methodologies
### Data Sources
- **Congressional Records**: Analyzed over 77,000 legislative documents.
- **Washington Representative Study**: Group-level variables like membership and lobbying expenditure.
- **Google Trends**: Public salience measures for policy areas.

### Analytical Framework
1. **Supervised Machine Learning**:
   - Classifies prominent mentions within debates (~81% accuracy with SVM).
2. **Generalized Linear Mixed-Effects Models**:
   - Examines factors influencing prominence at policy, group, and politician levels.
3. **Data Integration**:
   - Combines multiple data sources for robust analysis.

---

## Subfolder Summaries

### **1. Data Collection**
- **Purpose**: Scripts to gather legislative data, policy metadata, and congressional profiles.
- **Key Scripts**:
  - `1.govinfo_data_fetcher.py`: Collects legislative transcripts.
  - `7.fetch_bill_data.py`: Gathers metadata on bills.
  - `policy_salience_pipeline.py`: Analyzes public interest using Google Trends.
- **Outputs**:
  - Legislative transcripts, bill metadata, and salience metrics.

---

### **2. Data Processing**
- **Purpose**: Refines, cleans, and structures raw data.
- **Key Scripts**:
  - `process_api_results.py`: Processes API responses.
  - `assign_speakers_to_granules.py`: Links legislative text to speakers.
  - `policy_area_mapper.py`: Maps committees and topics to policy areas.
- **Outputs**:
  - Structured datasets and speaker-text mappings.

---

### **3. Supervised Learning Classifiers**
- **Purpose**: Builds and evaluates machine learning models for classifying prominence.
- **Key Script**:
  - `text_classifier_pipeline.py`: Handles preprocessing, training, evaluation, and labeling of data.
- **Outputs**:
  - Classification reports, labeled datasets, and predictions.

---

### **4. Modeling and Analysis**
- **Purpose**: Analyzes factors influencing prominence through advanced statistical models.
- **Key Scripts**:
  - `model_policy_salience.R`: Explores the role of public salience in prominence.
  - `DataProcessingAndRegression.py`: Prepares data for regression and modeling.
- **Outputs**:
  - Regression models, cleaned datasets, and statistical summaries.

---

### **5. Visualization and Reporting**
- **Purpose**: Visualizes findings and compiles reports for academic and policy audiences.
- **Key Documents**:
  - `Technical Report MA Thesis.pdf`: Summarizes methods and findings.
  - `Thesis_UvA_Kaleb_Mazurek.pdf`: Comprehensive thesis document.
- **Insights**:
  - Advocacy prominence is shaped by lobbying, policy engagement, and niche focus.
  - Seniority in politicians and high-salience issues may not guarantee greater visibility.

---

## Key Findings
- **Policy Salience**:
  - Medium-salience issues increase prominence, while high-salience areas show mixed effects.
- **Group Characteristics**:
  - Lobbying expenditure and broad policy engagement are significant predictors of prominence.
- **Speaker Attributes**:
  - Seniority and election-year status exhibit unexpected negative or mixed effects.

---

## Prerequisites
1. **Python**: Install dependencies with:
   ```bash
   pip install -r requirements.txt
   ```
2. **R**: Install required libraries:
   ```R
   install.packages(c("lme4", "ggplot2", "broom.mixed"))
   ```
3. **API Keys**: Obtain keys for GovInfo, Congress API, and Google Trends.

---

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/username/MastersThesis_InterestGroupAnalysis.git
   ```
2. Navigate to subfolders and review their README files for script-specific instructions.
3. Run scripts sequentially for each stage:
   - Data Collection → Data Processing → Machine Learning → Modeling → Visualization.

---

## Contribution
Contributions are welcome! Fork the repository and submit pull requests for improvements.

---

## Citation
If using this repository for your research, please cite:

Mazurek, Kaleb. *Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence*. University of Amsterdam, 2023.

---

## License
This repository is licensed under the MIT License. See `LICENSE` for details.

--- 

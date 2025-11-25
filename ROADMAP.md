# Project Roadmap: Reproducible Interest Group Analysis Pipeline

**Last Updated**: November 25, 2024
**Project Status**: Modernization Phase
**Timeline**: November 2024 - Q3 2025

---

## Vision Statement

> Transform a successful but non-reproducible thesis project into a **fully reproducible, modular, and extensible research pipeline** for analyzing interest group prominence in legislative debates.

### Core Goals

1. **Transparency**: Honestly document the research journey, including limitations
2. **Reproducibility**: Enable complete reproduction of analysis from validated datasets
3. **Extensibility**: Make it easy to collect and analyze new Congressional Record data
4. **Education**: Serve as a template for computational social science best practices
5. **Impact**: Enable other researchers to build on this methodology

---

## Timeline Overview

```
2024 Q4 (Nov-Dec)          2025 Q1 (Jan-Mar)          2025 Q2 (Apr-Jun)          2025 Q3 (Jul-Sep)
     │                          │                          │                          │
     ├─ Phase 1 ✅              ├─ Phase 3                 ├─ Phase 4                 ├─ Phase 5
     │  Restructuring           │  Pipeline Dev            │  Testing/Validation     │  Extension
     │                          │                          │                          │
     ├─ Phase 2                 │                          │                          └─ Publication
     │  Documentation           │                          │
     │                          │                          │
     └────────────────────────────────────────────────────────────────────────────────────────────>
      Repository                Modular                    Automated                New Congress
      Modernization             Components                 Testing                  Data (118th)
```

---

## Phase 1: Repository Restructuring ✅

**Timeline**: November 15-25, 2024
**Status**: COMPLETE

### Objectives

- [x] Archive legacy code in `/legacy/` folder
- [x] Create new directory structure
- [x] Write comprehensive documentation explaining the transition
- [x] Establish the `multi_level_data` dataset as reproducibility anchor

### Deliverables

- [x] `/legacy/` folder with all original code preserved
- [x] `/legacy/README.md` documenting why code is not reproducible
- [x] New top-level `README.md` with project overview
- [x] `MIGRATION_PLAN.md` with detailed transition strategy
- [x] `ROADMAP.md` (this document) with future timeline
- [x] Empty directory structure for future development

### Outcomes

✅ **Repository is now organized and transparent**
- Clear separation between legacy and modern code
- Honest documentation of limitations
- Foundation for future development established

---

## Phase 2: Comprehensive Documentation

**Timeline**: December 1-31, 2024
**Status**: PLANNED

### Objectives

- [ ] Fully document the `multi_level_data` dataset
- [ ] Create complete data dictionary with variable definitions
- [ ] Write step-by-step reproducibility guide
- [ ] Document API setup procedures
- [ ] Establish data versioning guidelines

### Deliverables

#### 2.1 Data Documentation

- [ ] **`docs/data_dictionary.md`**
  - Complete list of all variables in `multi_level_data`
  - Data types, valid ranges, and missing data patterns
  - Source and derivation for each variable
  - Examples of typical values

- [ ] **`docs/multi_level_data_specification.md`**
  - Hierarchical structure (mentions → paragraphs → documents)
  - File format and schema
  - Loading instructions for Python/R
  - Data quality notes and known limitations

- [ ] **`data/README.md`**
  - Data organization philosophy
  - Versioning strategy (using dates or semantic versioning)
  - Guidelines for adding new datasets
  - Instructions for large file storage (if needed)

- [ ] **`data/processed/multi_level_data/README.md`**
  - Specific documentation for the anchor dataset
  - Generation process and date
  - Validation checks performed
  - Citation information

#### 2.2 Reproducibility Guides

- [ ] **`docs/reproducibility_notes.md`**
  - Step-by-step guide to reproducing thesis results
  - Software requirements and versions
  - Expected runtime and computational resources
  - Troubleshooting common issues
  - Known deviations from original results

- [ ] **`docs/api_setup_guide.md`**
  - How to obtain GovInfo API key
  - How to access Congress.gov API
  - Rate limiting and best practices
  - Example API calls with expected responses

#### 2.3 Pipeline Documentation

- [ ] **`pipeline/README.md`**
  - Overview of pipeline architecture
  - Design principles and decisions
  - Module dependencies and data flow
  - How to run individual stages vs. full pipeline

- [ ] READMEs for each pipeline subfolder:
  - [ ] `pipeline/01_data_collection/README.md`
  - [ ] `pipeline/02_data_processing/README.md`
  - [ ] `pipeline/03_machine_learning/README.md`
  - [ ] `pipeline/04_analysis/README.md`
  - [ ] `pipeline/utils/README.md`

#### 2.4 Supporting Documentation

- [ ] **`outputs/README.md`** - How to interpret outputs
- [ ] **`notebooks/README.md`** - Exploratory analysis guidelines
- [ ] **`tests/README.md`** - Testing philosophy and how to run tests

### Success Criteria

✅ A researcher unfamiliar with the project can:
- Understand the dataset structure without reading code
- Reproduce thesis findings using `multi_level_data`
- Set up API credentials to collect new data
- Navigate the repository without confusion

---

## Phase 3: Modular Pipeline Development

**Timeline**: January - March 2025
**Status**: PLANNED

### Objectives

- [ ] Extract reusable functions from legacy code
- [ ] Build modular, testable data collection scripts
- [ ] Implement validated data processing pipeline
- [ ] Recreate machine learning classification workflow
- [ ] Develop analysis and visualization modules

### Deliverables by Month

#### January 2025: Foundation

- [ ] **Environment Setup**
  - `requirements.txt` with pinned Python packages
  - `environment.yml` for conda users
  - `Dockerfile` for containerized reproducibility
  - `docker-compose.yml` for multi-service orchestration

- [ ] **Utilities Module** (`pipeline/utils/`)
  - `config.py` - Configuration management
  - `logging_config.py` - Structured logging
  - `validators.py` - Data quality checks
  - `file_io.py` - Standardized file operations
  - `api_client.py` - Base class for API interactions

#### February 2025: Data Collection & Processing

- [ ] **Data Collection Module** (`pipeline/01_data_collection/`)
  - `fetch_congressional_records.py` - GovInfo API client
  - `fetch_member_profiles.py` - Congress.gov member data
  - `fetch_bill_metadata.py` - Bill information
  - `collect_policy_salience.py` - Google Trends integration
  - `config/` - API endpoints, parameters, mappings

- [ ] **Data Processing Module** (`pipeline/02_data_processing/`)
  - `parse_congressional_record.py` - HTML/XML parsing
  - `assign_speakers.py` - Speaker identification
  - `extract_mentions.py` - Interest group mention detection
  - `remove_duplicates.py` - TF-IDF duplicate filtering
  - `map_policy_areas.py` - Policy area categorization

#### March 2025: Machine Learning & Analysis

- [ ] **Machine Learning Module** (`pipeline/03_machine_learning/`)
  - `prepare_training_data.py` - Feature engineering
  - `train_classifier.py` - SVM/NB/RF training
  - `generate_predictions.py` - Apply trained models
  - `evaluate_models.py` - Performance metrics
  - `models/` - Serialized model artifacts

- [ ] **Analysis Module** (`pipeline/04_analysis/`)
  - `integrate_datasets.py` - Merge all data sources
  - `descriptive_statistics.py` - Summary tables
  - `run_multilevel_models.R` - GLMM in R
  - `generate_visualizations.py` - Figures for publication
  - `export_results.py` - Format for reports

### Code Quality Standards

All code must:
- ✅ Pass `black` formatting
- ✅ Pass `flake8` linting
- ✅ Have comprehensive docstrings (Google style)
- ✅ Include type hints where appropriate
- ✅ Have no hard-coded paths or credentials
- ✅ Use configuration files for parameters

### Success Criteria

✅ Pipeline can run end-to-end on a sample dataset
✅ Each module has clear inputs and outputs
✅ Functions are reusable across different Congress sessions
✅ Code is self-documenting with good naming conventions

---

## Phase 4: Testing and Validation

**Timeline**: April - June 2025
**Status**: PLANNED

### Objectives

- [ ] Write comprehensive unit tests for all functions
- [ ] Implement integration tests for full pipeline
- [ ] Validate new outputs against `multi_level_data`
- [ ] Set up continuous integration (CI)
- [ ] Achieve 80%+ test coverage

### Deliverables by Month

#### April 2025: Unit Testing

- [ ] **Test Suite** (`tests/`)
  - `test_data_collection.py` - API client tests
  - `test_data_processing.py` - Parsing and transformation tests
  - `test_machine_learning.py` - Classifier tests
  - `test_analysis.py` - Statistical computation tests
  - `test_utils.py` - Utility function tests
  - `conftest.py` - Pytest fixtures and configuration

- [ ] **Test Coverage**
  - Set up `pytest-cov` for coverage reporting
  - Achieve 60%+ coverage initially
  - Target 80%+ coverage by end of month

#### May 2025: Integration Testing

- [ ] **Integration Tests** (`tests/integration/`)
  - `test_full_pipeline.py` - End-to-end workflow
  - `test_data_quality.py` - Output validation
  - `test_error_handling.py` - Failure modes

- [ ] **Validation Notebooks** (`notebooks/validation/`)
  - `01_compare_speaker_assignments.ipynb`
  - `02_compare_mention_extraction.ipynb`
  - `03_compare_classifier_outputs.ipynb`
  - `04_compare_statistical_models.ipynb`

- [ ] **Validation Reports**
  - Document differences between legacy and new pipeline
  - Classify differences as acceptable vs. concerning
  - Explain sources of variance

#### June 2025: CI/CD and Containerization

- [ ] **Continuous Integration** (`.github/workflows/`)
  - `tests.yml` - Run tests on push/PR
  - `lint.yml` - Code quality checks
  - `build-docker.yml` - Build and test Docker image

- [ ] **Containerization**
  - Finalize `Dockerfile` with all dependencies
  - Test Docker build on Linux/Mac/Windows
  - Document Docker usage in README
  - Create Docker image on Docker Hub (optional)

- [ ] **Performance Benchmarking**
  - Measure runtime for each pipeline stage
  - Identify bottlenecks
  - Optimize slow components

### Success Criteria

✅ 80%+ test coverage across all modules
✅ All tests pass in CI environment
✅ Docker container can run full pipeline
✅ New outputs match `multi_level_data` within documented tolerances
✅ Performance is acceptable (full pipeline < 24 hours on standard hardware)

---

## Phase 5: Extension to New Congress Sessions

**Timeline**: July - September 2025
**Status**: PLANNED

### Objectives

- [ ] Collect data for 118th Congress (2024-2025)
- [ ] Validate that pipeline generalizes to new time periods
- [ ] Compare patterns with 2014-2018 baseline
- [ ] Publish findings and extended dataset

### Deliverables by Month

#### July 2025: Data Collection

- [ ] **Collect 118th Congress Data**
  - Run data collection for 2024 (partial year)
  - Validate API responses match expected structure
  - Store raw data with proper version control

- [ ] **Initial Processing**
  - Apply processing pipeline to new data
  - Check for unexpected issues (format changes, new edge cases)
  - Document any manual corrections needed

#### August 2025: Analysis

- [ ] **Classification and Modeling**
  - Apply trained classifier to new mentions
  - Run multilevel models on extended dataset
  - Compare coefficients with 2014-2018 results

- [ ] **Exploratory Analysis** (`notebooks/analysis/`)
  - `05_temporal_trends_2014-2024.ipynb`
  - `06_policy_area_shifts.ipynb`
  - `07_group_prominence_changes.ipynb`
  - `08_partisan_differences.ipynb`

- [ ] **Validation**
  - Check for data quality issues
  - Identify interesting substantive patterns
  - Decide if findings warrant publication

#### September 2025: Publication and Dissemination

- [ ] **Research Outputs**
  - Write technical report on extended findings
  - Create visualizations for blog post or paper
  - Prepare dataset for public release (if applicable)

- [ ] **Documentation Updates**
  - Update READMEs with new Congress data
  - Document lessons learned from extension
  - Create video tutorial for using the pipeline

- [ ] **Community Engagement**
  - Present at computational social science conference
  - Write blog post on reproducible research practices
  - Solicit feedback from other researchers

### Success Criteria

✅ Pipeline runs on new data without major modifications
✅ New dataset has comparable quality to original
✅ Substantive findings are interpretable and interesting
✅ Extended dataset is publicly available (if permitted)
✅ At least one publication or presentation results

---

## Future Enhancements (Post-2025)

### Potential Extensions

#### 1. **Interactive Web Dashboard**
- Shiny app or Dash dashboard for exploring patterns
- Filter by time period, policy area, or group type
- Download custom subsets of data

#### 2. **Automated Data Updates**
- Scheduled GitHub Actions to fetch new data monthly
- Automatically process and classify new mentions
- Generate updated reports

#### 3. **Alternative Classifiers**
- Fine-tune large language models (BERT, RoBERTa)
- Compare performance to SVM baseline
- Investigate zero-shot classification approaches

#### 4. **Network Analysis**
- Co-mention networks of interest groups
- Legislative collaboration networks
- Topic co-occurrence networks

#### 5. **Causal Inference**
- Difference-in-differences for policy shocks
- Instrumental variables for lobbying effects
- Synthetic control for case studies

#### 6. **API Service**
- REST API for programmatic access to data
- Query mentions by group, politician, policy area
- Rate-limited public access with API keys

---

## Risk Management

### Potential Obstacles

#### 1. **Time Constraints**
**Risk**: Balancing project with other commitments

**Mitigation**:
- Break work into small, manageable tasks
- Focus on minimum viable deliverables
- Be willing to extend timeline if needed

#### 2. **API Changes**
**Risk**: GovInfo or Congress.gov APIs change structure

**Mitigation**:
- Monitor API documentation for changes
- Build adapters to handle multiple versions
- Cache raw responses for historical reference

#### 3. **Validation Failures**
**Risk**: New pipeline produces different results than legacy

**Mitigation**:
- Document acceptable tolerance levels upfront
- Investigate differences systematically
- Prioritize substantive match over exact replication

#### 4. **Technical Challenges**
**Risk**: Unforeseen complexity in refactoring legacy code

**Mitigation**:
- Start with simplest components first
- Ask for help on Stack Overflow / GitHub discussions
- Consider hiring a consultant for specific issues

---

## Success Metrics

### Quantitative Indicators

| Metric | Target | Current |
|--------|--------|---------|
| Test coverage | 80%+ | 0% (baseline) |
| Documentation completeness | 100% of modules | 30% |
| Pipeline runtime (118th Congress) | < 24 hours | TBD |
| Data quality (speaker assignment accuracy) | > 90% | ~85% (legacy) |
| Code quality (linting) | 100% pass | TBD |

### Qualitative Indicators

✅ **Reproducibility**: Independent researcher can run pipeline
✅ **Transparency**: Limitations are honestly documented
✅ **Usability**: Clear documentation and examples
✅ **Extensibility**: Easy to add new Congress sessions
✅ **Impact**: Other researchers use or cite the work

---

## Call for Contributions

This is an **open science** project. We welcome contributions in the following areas:

### Areas Where Help Is Needed

1. **Testing**: Write unit tests for data processing functions
2. **Documentation**: Improve code comments and docstrings
3. **Validation**: Cross-check outputs against alternative methods
4. **Extension**: Collect data for state legislatures or other countries
5. **Analysis**: Run alternative statistical models
6. **Visualization**: Create interactive dashboards

### How to Get Involved

1. **Star the repository** on GitHub to stay updated
2. **Open an issue** to suggest improvements or report bugs
3. **Fork and submit a PR** with your contributions
4. **Share the project** with colleagues who might benefit

---

## Communication and Updates

### Regular Updates

- **Monthly**: Update this ROADMAP with progress
- **Quarterly**: Write blog post on development status
- **Major Milestones**: Announce on Twitter/LinkedIn/academic mailing lists

### Contact

**Project Lead**: Kaleb Mazurek
**Email**: kalebmazurek@gmail.com
**GitHub**: [@kmazurek95](https://github.com/kmazurek95)

For questions, suggestions, or collaboration inquiries:
- **GitHub Issues**: [kmazurek95/MastersThesis_InterestGroupAnalysis/issues](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)
- **Email**: kalebmazurek@gmail.com

---

## Conclusion

This roadmap represents an **ambitious but achievable plan** to transform a successful thesis project into a fully reproducible research pipeline.

### Key Principles

1. **Honesty over polish**: We document limitations transparently
2. **Progress over perfection**: We build incrementally
3. **Reusability over novelty**: We prioritize practical value
4. **Community over individual**: We invite collaboration

### Long-Term Vision

By Q3 2025, this repository will serve as:

- ✅ A **template** for reproducible computational social science
- ✅ A **resource** for researchers studying interest groups
- ✅ A **case study** in honest research documentation
- ✅ A **platform** for ongoing data collection and analysis

---

**Roadmap Version**: 1.0
**Last Updated**: November 25, 2024
**Next Review**: December 15, 2024
**Status**: Phase 1 Complete, Phase 2 Beginning

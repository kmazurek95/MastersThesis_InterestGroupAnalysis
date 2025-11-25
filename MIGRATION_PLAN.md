# Migration Plan: Legacy to Modern Reproducible Pipeline

**Document Version**: 1.0
**Date**: November 2024
**Status**: Implementation in progress

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Migration Philosophy](#migration-philosophy)
3. [Current State Assessment](#current-state-assessment)
4. [Target State Vision](#target-state-vision)
5. [Migration Strategy](#migration-strategy)
6. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
7. [The Role of `multi_level_data`](#the-role-of-multi_level_data)
8. [Validation and Testing Strategy](#validation-and-testing-strategy)
9. [Risk Management](#risk-management)
10. [Success Metrics](#success-metrics)

---

## Executive Summary

### The Challenge

The original thesis repository (2023) contains valuable research code that **successfully produced valid findings**, but is **not fully reproducible** due to:

- Manual data processing steps not captured in code
- Missing intermediate datasets
- Ad-hoc exploratory analysis
- Undocumented parameter choices
- Inconsistent file formats and naming conventions

### The Solution

Rather than **pretending the old system was reproducible**, we:

1. **Archive** the original code in `/legacy/` with honest documentation
2. **Establish** `multi_level_data` as the validated ground truth
3. **Rebuild** a modern, modular, reproducible pipeline in `/pipeline/`
4. **Document** the transition process transparently

### Timeline

- **Phase 1 (November 2024)**: Repository restructuring ✅ Current
- **Phase 2 (December 2024)**: Core documentation and data specification
- **Phase 3 (Q1 2025)**: Modular pipeline development
- **Phase 4 (Q2 2025)**: Validation and testing
- **Phase 5 (Q3 2025)**: Extension to new Congress sessions

---

## Migration Philosophy

### Core Principles

#### 1. **Transparency Over Perfection**

We acknowledge that research is iterative and messy. **Honest documentation** of limitations is more valuable than presenting a false narrative of reproducibility.

> "It is better to be vaguely right than exactly wrong." - Carveth Read

#### 2. **Preservation of Research History**

The legacy code represents **real research decisions** and should be preserved exactly as it existed, not rewritten to appear more polished.

#### 3. **Progressive Enhancement**

We don't need to reproduce **every step** of the original pipeline. Instead:

- Start with the validated `multi_level_data` dataset
- Rebuild only what's necessary for future extensions
- Focus on reproducibility **going forward**

#### 4. **Practical Reusability**

The goal is **practical reproducibility** for:

- Extending the analysis to new Congress sessions
- Allowing other researchers to replicate the methodology
- Teaching computational social science best practices

---

## Current State Assessment

### What Works (Preserved in Legacy)

✅ **Data Collection Scripts**
- GovInfo API fetching logic is solid
- Congress.gov API integration works
- Google Trends data pipeline is functional

✅ **NLP Processing**
- Speaker assignment heuristics are reasonable
- Interest group extraction patterns are well-defined
- TF-IDF duplicate detection is mathematically sound

✅ **Machine Learning Models**
- SVM classifier achieves 81% accuracy
- Training data labeling process is documented
- Feature engineering is well-motivated

✅ **Statistical Modeling**
- GLMM specifications are appropriate
- R code for multilevel models is correct
- Results are statistically valid

### What Doesn't Work (Requires Rebuild)

❌ **End-to-End Reproducibility**
- Cannot run scripts sequentially from raw data
- Intermediate files have hard-coded paths
- Some manual steps are not scripted

❌ **Data Provenance**
- Original raw data not version-controlled
- Transformations not fully documented
- No audit trail for manual corrections

❌ **Environment Management**
- No `requirements.txt` with pinned versions
- Python/R package versions not documented
- Local dependencies not specified

❌ **Modularity**
- Monolithic scripts with multiple responsibilities
- Code duplication across files
- No separation of configuration from logic

---

## Target State Vision

### Repository Structure

```
MastersThesis_InterestGroupAnalysis/
├── legacy/                    # Historical archive (read-only)
├── data/                      # Organized data hierarchy
│   ├── raw/                   # API outputs (not in git, documented)
│   ├── processed/             # Clean datasets (versioned)
│   ├── interim/               # Intermediate outputs (documented)
│   └── external/              # Third-party data (sourced)
├── pipeline/                  # Modern reproducible code
│   ├── 01_data_collection/    # Modular API scripts
│   ├── 02_data_processing/    # Validated transformations
│   ├── 03_machine_learning/   # Classifier training
│   ├── 04_analysis/           # Statistical models
│   └── utils/                 # Shared functions
├── docs/                      # Comprehensive documentation
├── outputs/                   # Analysis results
├── notebooks/                 # Exploratory analysis
└── tests/                     # Automated testing
```

### Key Capabilities

1. **One-Command Reproduction**
   ```bash
   python pipeline/run_full_pipeline.py --congress 118 --year 2024
   ```

2. **Modular Reusability**
   ```python
   from pipeline.utils.speaker_assignment import assign_speakers
   speakers = assign_speakers(paragraphs, members)
   ```

3. **Comprehensive Testing**
   ```bash
   pytest tests/ --cov=pipeline
   ```

4. **Environment Reproducibility**
   ```bash
   docker-compose up --build
   ```

---

## Migration Strategy

### What We're NOT Doing

❌ **Rewriting legacy code**: The original scripts stay as-is in `/legacy/`
❌ **Recreating lost data**: We won't try to regenerate unavailable raw data
❌ **Backwards compatibility**: New code doesn't need to match old outputs exactly
❌ **Complete reproduction**: We're building forward, not reconstructing backward

### What We ARE Doing

✅ **Building from `multi_level_data`**: Start with validated dataset
✅ **Modularizing components**: Extract reusable functions from legacy code
✅ **Adding documentation**: Explain every decision and assumption
✅ **Enabling extension**: Make it easy to add new Congress years
✅ **Implementing best practices**: Testing, version control, containerization

---

## Phase-by-Phase Implementation

### Phase 1: Repository Restructuring ✅ (November 2024)

**Goals**:
- Move original code to `/legacy/`
- Create new directory structure
- Write initial documentation

**Deliverables**:
- ✅ `/legacy/` folder with archived code
- ✅ `/legacy/README.md` documenting limitations
- ✅ New top-level `README.md` explaining transition
- ✅ Empty `/pipeline/`, `/data/`, `/docs/` folders
- ✅ `MIGRATION_PLAN.md` (this document)
- ✅ `ROADMAP.md` with future timeline

**Status**: COMPLETE

---

### Phase 2: Core Documentation (December 2024)

**Goals**:
- Document the `multi_level_data` dataset comprehensively
- Create data dictionaries and variable specifications
- Write reproducibility notes and API setup guides

**Deliverables**:
- [ ] `docs/data_dictionary.md` - All variable definitions
- [ ] `docs/multi_level_data_specification.md` - Dataset structure
- [ ] `docs/reproducibility_notes.md` - Step-by-step guide
- [ ] `docs/api_setup_guide.md` - GovInfo/Congress.gov setup
- [ ] `data/README.md` - Data organization and versioning
- [ ] `data/processed/multi_level_data/README.md` - Dataset documentation

**Success Criteria**:
- A researcher can understand the dataset without reading legacy code
- Variable names, types, and missing data patterns are fully documented
- Examples show how to load and analyze the data

---

### Phase 3: Modular Pipeline Development (Q1 2025)

**Goals**:
- Extract reusable functions from legacy code
- Build modular data collection and processing scripts
- Implement proper error handling and logging

**Deliverables**:

#### 3.1 Data Collection Module
- [ ] `pipeline/01_data_collection/fetch_congressional_records.py`
- [ ] `pipeline/01_data_collection/fetch_member_profiles.py`
- [ ] `pipeline/01_data_collection/fetch_bill_metadata.py`
- [ ] `pipeline/01_data_collection/collect_policy_salience.py`
- [ ] Configuration files for API endpoints and parameters

#### 3.2 Data Processing Module
- [ ] `pipeline/02_data_processing/parse_congressional_record.py`
- [ ] `pipeline/02_data_processing/assign_speakers.py`
- [ ] `pipeline/02_data_processing/extract_mentions.py`
- [ ] `pipeline/02_data_processing/remove_duplicates.py`
- [ ] `pipeline/02_data_processing/map_policy_areas.py`

#### 3.3 Machine Learning Module
- [ ] `pipeline/03_machine_learning/prepare_training_data.py`
- [ ] `pipeline/03_machine_learning/train_classifier.py`
- [ ] `pipeline/03_machine_learning/generate_predictions.py`
- [ ] `pipeline/03_machine_learning/evaluate_models.py`

#### 3.4 Analysis Module
- [ ] `pipeline/04_analysis/integrate_datasets.py`
- [ ] `pipeline/04_analysis/descriptive_statistics.py`
- [ ] `pipeline/04_analysis/run_multilevel_models.R`
- [ ] `pipeline/04_analysis/generate_visualizations.py`

#### 3.5 Utilities
- [ ] `pipeline/utils/config.py` - Configuration management
- [ ] `pipeline/utils/logging_config.py` - Structured logging
- [ ] `pipeline/utils/validators.py` - Data quality checks
- [ ] `pipeline/utils/file_io.py` - Standardized file operations

**Success Criteria**:
- Each module has clear inputs and outputs
- Functions have comprehensive docstrings
- Code passes linting and style checks (black, flake8)
- No hard-coded paths or magic numbers

---

### Phase 4: Validation and Testing (Q2 2025)

**Goals**:
- Validate new pipeline outputs against `multi_level_data`
- Write unit tests for all functions
- Implement integration tests for full pipeline
- Create continuous integration workflows

**Deliverables**:

#### 4.1 Unit Tests
- [ ] `tests/test_data_collection.py`
- [ ] `tests/test_data_processing.py`
- [ ] `tests/test_machine_learning.py`
- [ ] `tests/test_analysis.py`
- [ ] `tests/test_utils.py`

#### 4.2 Integration Tests
- [ ] `tests/integration/test_full_pipeline.py`
- [ ] `tests/integration/test_data_quality.py`

#### 4.3 Validation Notebooks
- [ ] `notebooks/validation/compare_speaker_assignments.ipynb`
- [ ] `notebooks/validation/compare_mention_extraction.ipynb`
- [ ] `notebooks/validation/compare_classifier_outputs.ipynb`

#### 4.4 CI/CD
- [ ] `.github/workflows/tests.yml` - Run tests on push
- [ ] `.github/workflows/lint.yml` - Code quality checks
- [ ] `Dockerfile` and `docker-compose.yml` for reproducibility

**Success Criteria**:
- 80%+ test coverage for pipeline code
- All tests pass in CI environment
- Docker container successfully runs full pipeline
- New pipeline outputs match `multi_level_data` within acceptable tolerances

---

### Phase 5: Extension to New Congress (Q3 2025)

**Goals**:
- Use new pipeline to collect 2024-2025 Congressional Record data
- Validate that pipeline generalizes to new time periods
- Document any necessary adaptations

**Deliverables**:
- [ ] Collect 118th Congress (2024-2025) data
- [ ] Process and classify new mentions
- [ ] Compare patterns with 2014-2018 data
- [ ] Publish extended dataset (if permitted)
- [ ] Write blog post or technical report on findings

**Success Criteria**:
- Pipeline runs without modification on new data
- New dataset has similar structure to `multi_level_data`
- Results are substantively interpretable

---

## The Role of `multi_level_data`

### What Is It?

The `multi_level_data` dataset is a **cleaned, validated, hierarchical file** containing:

- All processed Congressional Record documents (2014-2018)
- Speaker assignments and metadata
- Interest group mentions with prominence labels
- Machine learning predictions
- Integrated group-level and policy-level variables

### Why Is It Critical?

1. **Ground Truth**: Validated outputs from the original thesis pipeline
2. **Reproducibility Anchor**: Starting point for modern analysis
3. **Validation Target**: New pipeline should produce similar results
4. **Documentation Aid**: Concrete example of expected data structure

### How We Use It

#### For Analysis (Now)
Researchers can immediately use `multi_level_data` to:
- Reproduce thesis findings
- Run alternative statistical models
- Test new hypotheses
- Visualize patterns

#### For Validation (Future)
When building the new pipeline, we:
- Compare new outputs to `multi_level_data`
- Identify systematic differences
- Document acceptable tolerances
- Flag breaking changes

#### For Extension (Later)
When collecting new data:
- Use `multi_level_data` structure as template
- Apply same variable naming conventions
- Maintain compatibility for longitudinal analysis

---

## Validation and Testing Strategy

### Three Levels of Validation

#### 1. **Unit-Level Validation**
Test individual functions in isolation:

```python
def test_assign_speaker():
    """Test that speaker assignment identifies correct member."""
    paragraph = "Mr. SMITH. Thank you for yielding."
    members = [{'bioguide_id': 'S000001', 'last_name': 'SMITH'}]
    result = assign_speaker(paragraph, members)
    assert result['bioguide_id'] == 'S000001'
```

#### 2. **Integration Validation**
Test that pipeline stages connect correctly:

```python
def test_data_collection_to_processing():
    """Test that raw data parses correctly."""
    raw_data = fetch_congressional_record(congress=114, year=2015)
    processed = parse_congressional_record(raw_data)
    assert 'paragraphs' in processed
    assert 'speaker_id' in processed['paragraphs'][0]
```

#### 3. **Output Validation**
Compare new pipeline outputs to `multi_level_data`:

```python
def test_mention_extraction_accuracy():
    """Test that mention extraction matches legacy."""
    legacy_mentions = load_multi_level_data()['mentions']
    new_mentions = extract_mentions(paragraphs)

    # Allow 95% match rate (some variance expected)
    match_rate = calculate_match_rate(legacy_mentions, new_mentions)
    assert match_rate >= 0.95
```

### Acceptable Tolerance Levels

| Component | Exact Match Required | Acceptable Variance |
|-----------|---------------------|---------------------|
| Speaker assignment | No | ±5% accuracy |
| Mention extraction (recall) | No | ±10% |
| Mention extraction (precision) | No | ±10% |
| Duplicate detection | No | ±5% similarity threshold |
| Classifier predictions | No | ±2% accuracy |
| Statistical model coefficients | No | ±0.05 standard errors |

---

## Risk Management

### Risk 1: Lost Institutional Knowledge

**Risk**: Original researcher may forget implementation details over time.

**Mitigation**:
- Document decisions **as soon as possible** while memory is fresh
- Create detailed code comments in legacy archive
- Record video walkthroughs of original workflow

### Risk 2: API Changes

**Risk**: GovInfo or Congress.gov APIs may change structure/authentication.

**Mitigation**:
- Document current API versions explicitly
- Build adapters to handle multiple API versions
- Cache raw API responses for future reference

### Risk 3: Scope Creep

**Risk**: Attempting to reproduce **everything** leads to never finishing.

**Mitigation**:
- Use `multi_level_data` as the stopping point
- Focus on **forward reproducibility** not backward reconstruction
- Accept that some legacy steps remain undocumented

### Risk 4: Validation Failure

**Risk**: New pipeline produces substantively different results.

**Mitigation**:
- Start with small test cases (single document)
- Validate incrementally at each stage
- Document known differences with explanations
- If differences are large, investigate systematically

---

## Success Metrics

### Phase Completion Criteria

Each phase is **complete** when:

✅ All deliverables are finished
✅ Documentation is comprehensive
✅ Code passes quality checks
✅ Peer review (if applicable) is complete

### Project Success Indicators

The migration is **successful** if:

1. **Reproducibility**: A new researcher can run the pipeline on 118th Congress data without assistance
2. **Transparency**: The transition from legacy to modern is fully documented
3. **Usability**: The `multi_level_data` dataset enables new analyses
4. **Extensibility**: The pipeline architecture supports future enhancements
5. **Credibility**: The honest acknowledgment of limitations increases research trustworthiness

---

## Next Steps

### Immediate Actions (This Week)

1. ✅ Complete repository restructuring
2. ✅ Write core documentation (this document)
3. [ ] Create `docs/multi_level_data_specification.md`
4. [ ] Write `data/README.md` with versioning guidelines

### Short-Term Actions (This Month)

1. [ ] Complete Phase 2 deliverables (documentation)
2. [ ] Set up development environment (Docker, requirements.txt)
3. [ ] Create GitHub project board for tracking progress
4. [ ] Write initial unit test examples

### Medium-Term Actions (Next Quarter)

1. [ ] Begin Phase 3 (modular pipeline development)
2. [ ] Refactor legacy code into modular functions
3. [ ] Implement data collection module
4. [ ] Start data processing module

---

## Conclusion

This migration represents a **honest, practical approach** to improving research reproducibility:

- We **preserve** the original work transparently
- We **acknowledge** limitations honestly
- We **build forward** with modern best practices
- We **enable** future research extensions

By using `multi_level_data` as the reproducibility anchor, we avoid the trap of trying to reconstruct every step of an exploratory research process while still providing a solid foundation for future work.

---

**Document Maintainer**: Kaleb Mazurek (kalebmazurek@gmail.com)
**Last Updated**: November 25, 2024
**Next Review**: December 15, 2024

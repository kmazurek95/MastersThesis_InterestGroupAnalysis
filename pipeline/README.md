# Modern Reproducible Pipeline

**Purpose**: Modular, testable, and documented data collection and analysis pipeline

**Status**: Under Development (Phase 3 planned for Q1 2025)

---

## Overview

This directory contains a **completely rebuilt pipeline** for collecting, processing, and analyzing Congressional Record data. Unlike the legacy code, this pipeline is:

✅ **Modular**: Each component has a single, well-defined responsibility
✅ **Testable**: All functions have unit tests
✅ **Documented**: Comprehensive docstrings and README files
✅ **Reproducible**: No manual steps, all parameters in config files
✅ **Extensible**: Easy to add new Congress sessions or policy areas

---

## Pipeline Architecture

```
Input: GovInfo API + Congress.gov API + External Data
   │
   ├─> 01_data_collection/       [Fetch raw data from APIs]
   │       └─> Output: data/raw/
   │
   ├─> 02_data_processing/        [Parse, clean, transform]
   │       └─> Output: data/interim/ and data/processed/
   │
   ├─> 03_machine_learning/       [Train classifiers, predict prominence]
   │       └─> Output: outputs/models/ and data/processed/
   │
   └─> 04_analysis/               [Statistical models, visualizations]
           └─> Output: outputs/figures/ and outputs/tables/
```

---

## Directory Structure

```
pipeline/
├── README.md                      # This file
├── 01_data_collection/            # API clients and data fetching
│   ├── README.md
│   ├── fetch_congressional_records.py
│   ├── fetch_member_profiles.py
│   ├── fetch_bill_metadata.py
│   ├── collect_policy_salience.py
│   └── config/                    # API endpoints, parameters
│
├── 02_data_processing/            # Parsing and transformation
│   ├── README.md
│   ├── parse_congressional_record.py
│   ├── assign_speakers.py
│   ├── extract_mentions.py
│   ├── remove_duplicates.py
│   └── map_policy_areas.py
│
├── 03_machine_learning/           # Classifier training and prediction
│   ├── README.md
│   ├── prepare_training_data.py
│   ├── train_classifier.py
│   ├── generate_predictions.py
│   ├── evaluate_models.py
│   └── models/                    # Serialized model artifacts
│
├── 04_analysis/                   # Statistical modeling and visualization
│   ├── README.md
│   ├── integrate_datasets.py
│   ├── descriptive_statistics.py
│   ├── run_multilevel_models.R
│   └── generate_visualizations.py
│
├── utils/                         # Shared utilities
│   ├── README.md
│   ├── config.py                  # Configuration management
│   ├── logging_config.py          # Logging setup
│   ├── validators.py              # Data quality checks
│   ├── file_io.py                 # File operations
│   └── api_client.py              # Base API client class
│
└── run_full_pipeline.py           # Orchestrate entire workflow
```

---

## Usage

### Quick Start (When Implemented)

#### Option 1: Run Full Pipeline

Collect and process new Congressional Record data:

```bash
# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys (see docs/api_setup_guide.md)
export GOVINFO_API_KEY="your_key_here"
export CONGRESS_API_KEY="your_key_here"

# Run full pipeline
python pipeline/run_full_pipeline.py --congress 118 --year 2024
```

#### Option 2: Run Individual Stages

Execute specific pipeline stages:

```bash
# Stage 1: Data Collection
python pipeline/01_data_collection/fetch_congressional_records.py \
    --congress 118 --year 2024 --output data/raw/

# Stage 2: Data Processing
python pipeline/02_data_processing/parse_congressional_record.py \
    --input data/raw/govinfo_118_2024_crec.json \
    --output data/interim/parsed_crec_118_2024.parquet

# Stage 3: Machine Learning
python pipeline/03_machine_learning/generate_predictions.py \
    --input data/interim/extracted_mentions_118_2024.parquet \
    --model outputs/models/svm_classifier_v1.0.pkl \
    --output data/processed/classified_mentions_118_2024.parquet

# Stage 4: Analysis
Rscript pipeline/04_analysis/run_multilevel_models.R \
    --input data/processed/classified_mentions_118_2024.parquet \
    --output outputs/tables/
```

#### Option 3: Use as Library

Import and use pipeline functions in your own code:

```python
from pipeline.utils.config import load_config
from pipeline.data_processing.assign_speakers import assign_speakers_to_paragraphs
from pipeline.machine_learning.train_classifier import train_svm_classifier

# Load configuration
config = load_config('pipeline/config.yaml')

# Use modular functions
paragraphs = load_paragraphs('data/interim/parsed_crec.parquet')
members = load_members('data/external/congress_members.csv')
paragraphs_with_speakers = assign_speakers_to_paragraphs(paragraphs, members)

# Train classifier
classifier, metrics = train_svm_classifier(
    training_data='data/external/hand_labeled_mentions.csv',
    test_size=0.2,
    random_state=42
)
```

---

## Design Principles

### 1. Modularity

**Goal**: Each script does one thing well

**Implementation**:
- Functions are small (< 50 lines when possible)
- Scripts have clear inputs and outputs
- No side effects or hidden state

**Example**:
```python
# Good: Single responsibility
def assign_speaker(paragraph_text: str, members: list) -> dict:
    """Assign speaker to a single paragraph."""
    pass

# Avoid: Multiple responsibilities
def process_everything(data):
    """Parse, assign speakers, extract mentions, classify, analyze."""
    pass
```

### 2. Configuration Over Hard-Coding

**Goal**: All parameters in config files, not buried in code

**Implementation**:
```yaml
# pipeline/config.yaml
data_collection:
  govinfo_api:
    base_url: "https://api.govinfo.gov/collections/"
    rate_limit: 10  # requests per second
  congress_api:
    base_url: "https://api.congress.gov/v3/"
    rate_limit: 100

data_processing:
  speaker_assignment:
    confidence_threshold: 0.8
  mention_extraction:
    min_mention_length: 3
    max_mention_length: 100
```

### 3. Validation at Every Step

**Goal**: Catch errors early with data quality checks

**Implementation**:
```python
from pipeline.utils.validators import validate_dataframe_schema

def process_data(df):
    # Validate input
    validate_dataframe_schema(
        df,
        required_columns=['granule_id', 'paragraph_text'],
        unique_columns=['granule_id']
    )

    # Process...

    # Validate output
    validate_dataframe_schema(
        result,
        required_columns=['granule_id', 'speaker_id', 'paragraph_text'],
        no_nulls=['speaker_id']
    )

    return result
```

### 4. Comprehensive Logging

**Goal**: Understand what the pipeline is doing and debug failures

**Implementation**:
```python
import logging
from pipeline.utils.logging_config import setup_logging

logger = setup_logging(__name__)

def fetch_data(congress, year):
    logger.info(f"Fetching data for Congress {congress}, year {year}")
    try:
        data = api_call()
        logger.info(f"Successfully fetched {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}", exc_info=True)
        raise
```

### 5. Type Hints and Documentation

**Goal**: Make code self-documenting

**Implementation**:
```python
from typing import List, Dict, Optional
import pandas as pd

def extract_mentions(
    paragraphs: pd.DataFrame,
    interest_groups: List[str],
    min_confidence: float = 0.7
) -> pd.DataFrame:
    """
    Extract interest group mentions from paragraphs.

    Args:
        paragraphs: DataFrame with columns ['paragraph_id', 'text']
        interest_groups: List of interest group names to search for
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        DataFrame with columns ['paragraph_id', 'mention', 'confidence']

    Raises:
        ValueError: If paragraphs DataFrame missing required columns

    Example:
        >>> paragraphs = pd.DataFrame({'paragraph_id': [1], 'text': ['The ACLU argues...']})
        >>> groups = ['ACLU', 'NRA']
        >>> mentions = extract_mentions(paragraphs, groups)
    """
    pass
```

---

## Code Style and Quality

### Style Guide

- **Python**: Follow [PEP 8](https://pep8.org/)
- **Formatter**: Use `black` (line length = 100)
- **Linter**: Use `flake8` with exceptions for line length
- **Type Checker**: Use `mypy` (when type hints are added)
- **Docstrings**: Use Google style

### Quality Checks

Before committing, run:

```bash
# Format code
black pipeline/

# Lint code
flake8 pipeline/ --max-line-length=100 --ignore=E203,W503

# Run tests
pytest tests/ --cov=pipeline --cov-report=html

# Type check (when implemented)
mypy pipeline/
```

### Pre-Commit Hooks

Install pre-commit hooks to automate quality checks:

```bash
pip install pre-commit
pre-commit install

# Manually run hooks
pre-commit run --all-files
```

---

## Testing Strategy

### Unit Tests

Test individual functions in isolation:

```python
# tests/test_data_processing.py
import pytest
from pipeline.data_processing.assign_speakers import assign_speaker

def test_assign_speaker_simple():
    """Test speaker assignment for simple case."""
    paragraph = "Mr. SMITH. Thank you for yielding."
    members = [{'bioguide_id': 'S000001', 'last_name': 'SMITH'}]

    result = assign_speaker(paragraph, members)

    assert result['bioguide_id'] == 'S000001'
    assert result['confidence'] > 0.9
```

### Integration Tests

Test that pipeline stages connect correctly:

```python
# tests/integration/test_full_pipeline.py
def test_data_collection_to_processing():
    """Test that raw data flows correctly into processing."""
    # Fetch small sample
    raw_data = fetch_congressional_records(congress=114, year=2015, limit=10)

    # Process sample
    processed = parse_congressional_record(raw_data)

    # Validate structure
    assert 'paragraphs' in processed
    assert len(processed['paragraphs']) > 0
```

### Validation Tests

Compare new pipeline outputs to `multi_level_data`:

```python
# tests/validation/test_against_legacy.py
def test_speaker_assignment_matches_legacy():
    """Validate speaker assignments match multi_level_data."""
    legacy_data = pd.read_parquet('data/processed/multi_level_data/multi_level_data_v1.0.parquet')
    new_assignments = run_speaker_assignment_pipeline()

    # Allow 95% match rate
    match_rate = calculate_match_rate(legacy_data, new_assignments, key='speaker_id')
    assert match_rate >= 0.95, f"Match rate {match_rate:.2%} below 95% threshold"
```

---

## Error Handling

### Principles

1. **Fail fast**: Validate inputs at function entry
2. **Informative errors**: Include context in exception messages
3. **Graceful degradation**: Log warnings for non-critical issues
4. **Retry logic**: Handle transient API failures

### Example

```python
import time
from typing import Optional

def fetch_with_retry(url: str, max_retries: int = 3, backoff: float = 2.0) -> Optional[dict]:
    """
    Fetch data from API with exponential backoff retry logic.

    Args:
        url: API endpoint
        max_retries: Maximum number of retry attempts
        backoff: Backoff multiplier (seconds)

    Returns:
        API response as dictionary, or None if all retries fail

    Raises:
        ValueError: If URL is malformed
    """
    if not url.startswith('http'):
        raise ValueError(f"Invalid URL: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully fetched {url}")
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
            else:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            return None
```

---

## Performance Considerations

### Parallelization

Use multiprocessing for CPU-bound tasks:

```python
from multiprocessing import Pool

def process_paragraph(paragraph):
    # CPU-intensive processing
    return result

with Pool(processes=4) as pool:
    results = pool.map(process_paragraph, paragraphs)
```

### Batch Processing

Process data in chunks to manage memory:

```python
CHUNK_SIZE = 10000

for chunk in pd.read_csv('large_file.csv', chunksize=CHUNK_SIZE):
    processed_chunk = process(chunk)
    append_to_output(processed_chunk)
```

### Caching

Cache expensive computations:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_member_info(bioguide_id: str) -> dict:
    """Fetch member info with caching."""
    return fetch_from_api(bioguide_id)
```

---

## Extending the Pipeline

### Adding a New Processing Step

1. **Create the script** in appropriate subfolder
2. **Write unit tests** in `tests/`
3. **Update README** in that subfolder
4. **Integrate** into `run_full_pipeline.py`
5. **Document** in this README

### Adding a New Data Source

1. **Create fetcher script** in `01_data_collection/`
2. **Document API** in `docs/api_setup_guide.md`
3. **Add to config** in `config.yaml`
4. **Write integration test**

---

## FAQ

### Q: Can I use parts of the pipeline independently?

**A**: Yes! All modules are designed to be reusable. Import functions as needed.

### Q: How do I add a new Congress session?

**A**: Run `python pipeline/run_full_pipeline.py --congress 119 --year 2025`

### Q: What if the API structure changes?

**A**: Update the parsing logic in `01_data_collection/`. Tests will catch breaking changes.

### Q: How do I reproduce the original thesis results?

**A**: Use the `multi_level_data` dataset directly. See `docs/reproducibility_notes.md`.

---

## Contributing

See `CONTRIBUTING.md` for guidelines (coming soon).

Key points:
- Write tests for all new code
- Follow code style guidelines
- Document your changes
- Update this README if adding new modules

---

## Contact

For questions about the pipeline architecture or implementation:

**Kaleb Mazurek**
Email: kalebmazurek@gmail.com
GitHub: [@kmazurek95](https://github.com/kmazurek95)

---

**Status**: Phase 3 (Q1 2025) - Under Development
**Last Updated**: November 25, 2024

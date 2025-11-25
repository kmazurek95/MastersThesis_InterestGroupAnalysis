# Tests Directory

**Purpose**: Automated testing for pipeline code

**Status**: Under Development (Phase 4 planned for Q2 2025)

---

## Overview

Comprehensive test suite including:
- **Unit tests**: Test individual functions
- **Integration tests**: Test pipeline stages
- **Validation tests**: Compare outputs to `multi_level_data`

---

## Structure

```
tests/
├── README.md                          # This file
├── conftest.py                        # Pytest configuration and fixtures
├── test_data_collection.py            # Unit tests for data collection
├── test_data_processing.py            # Unit tests for data processing
├── test_machine_learning.py           # Unit tests for ML module
├── test_analysis.py                   # Unit tests for analysis
├── test_utils.py                      # Unit tests for utilities
├── integration/                       # Integration tests
│   ├── test_full_pipeline.py
│   └── test_data_quality.py
├── validation/                        # Validation against legacy
│   ├── test_speaker_assignment.py
│   └── test_mention_extraction.py
└── fixtures/                          # Test data
    ├── sample_crec_response.json
    └── sample_members.csv
```

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_processing.py

# Run with coverage report
pytest tests/ --cov=pipeline --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Run in verbose mode
pytest tests/ -v
```

---

## Writing Tests

### Example Unit Test

```python
# tests/test_data_processing.py
import pytest
from pipeline.data_processing.assign_speakers import assign_speaker

def test_assign_speaker_simple():
    """Test speaker assignment for straightforward case."""
    paragraph = "Mr. SMITH. Thank you for yielding."
    members = [{'bioguide_id': 'S000001', 'last_name': 'SMITH'}]

    result = assign_speaker(paragraph, members)

    assert result['bioguide_id'] == 'S000001'
    assert result['confidence'] > 0.9

def test_assign_speaker_ambiguous():
    """Test speaker assignment when multiple members match."""
    paragraph = "Mr. SMITH. Thank you."
    members = [
        {'bioguide_id': 'S000001', 'last_name': 'SMITH', 'state': 'CA'},
        {'bioguide_id': 'S000002', 'last_name': 'SMITH', 'state': 'TX'}
    ]

    result = assign_speaker(paragraph, members)

    # Should handle ambiguity gracefully
    assert result['bioguide_id'] in ['S000001', 'S000002']
    assert result['confidence'] < 0.9  # Lower confidence for ambiguous
```

### Example Integration Test

```python
# tests/integration/test_full_pipeline.py
@pytest.mark.slow
def test_pipeline_end_to_end(sample_data):
    """Test full pipeline on sample data."""
    # Collect (mocked)
    raw_data = fetch_congressional_records(congress=114, year=2015, limit=10)

    # Process
    parsed = parse_congressional_record(raw_data)
    with_speakers = assign_speakers(parsed)
    mentions = extract_mentions(with_speakers)

    # Validate structure
    assert 'mention_id' in mentions.columns
    assert len(mentions) > 0
```

---

## Test Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| `01_data_collection/` | 80%+ |
| `02_data_processing/` | 90%+ |
| `03_machine_learning/` | 85%+ |
| `04_analysis/` | 75%+ |
| `utils/` | 95%+ |
| **Overall** | **80%+** |

---

## Fixtures

Shared test data in `conftest.py`:

```python
# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_paragraphs():
    """Sample paragraphs for testing."""
    return pd.DataFrame({
        'paragraph_id': [1, 2, 3],
        'text': [
            'Mr. SMITH. Thank you.',
            'The ACLU supports this.',
            'Ms. JONES. I agree.'
        ]
    })

@pytest.fixture
def sample_members():
    """Sample congress members for testing."""
    return [
        {'bioguide_id': 'S000001', 'last_name': 'SMITH'},
        {'bioguide_id': 'J000001', 'last_name': 'JONES'}
    ]
```

---

**Last Updated**: November 25, 2024

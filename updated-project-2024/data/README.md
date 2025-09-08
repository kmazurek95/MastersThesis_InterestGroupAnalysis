# Data Directory

## Structure
```
data/
├── raw/         # Original, unprocessed data files
├── processed/   # Cleaned and structured datasets
└── external/    # External reference data sources
```

## Data Guidelines
- **Raw data**: Keep original files unchanged for reproducibility
- **Processed data**: Store cleaned versions with clear naming conventions  
- **External data**: Reference datasets, APIs keys files, etc.
- **Large files**: Consider using Git LFS for files >100MB

## Security Note
- Never commit sensitive data (API keys, personal information)
- Use `.gitignore` to exclude large datasets if needed
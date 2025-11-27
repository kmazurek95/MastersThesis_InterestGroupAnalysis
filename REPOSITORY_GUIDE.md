# Repository Guide: How to Navigate This Project

**Quick reference for understanding the structure and purpose of this repository**

---

## Overview

This repository contains a **Master's thesis project** analyzing interest group prominence in U.S. Congressional debates. It's organized to maximize transparency and reproducibility.

---

## Start Here

### 🎯 For Most Users

**Want to explore the analysis?**
→ Go to [analysis/01_Exploratory_Prominence_Analysis.ipynb](analysis/01_Exploratory_Prominence_Analysis.ipynb)

**Need setup instructions?**
→ Read [USAGE_GUIDE.md](USAGE_GUIDE.md)

**Want to understand the research?**
→ See [README.md](README.md) for project overview

### 📚 For Researchers

**Reproducing thesis findings:**
1. Read [README.md](README.md) - Understand the project
2. Follow [USAGE_GUIDE.md](USAGE_GUIDE.md) - Set up environment
3. Run [analysis/01_Exploratory_Prominence_Analysis.ipynb](analysis/01_Exploratory_Prominence_Analysis.ipynb) - Explore data
4. Check [analysis/README.md](analysis/README.md) - Run all analyses

**Extending the research:**
1. Fork the repository (see [CONTRIBUTING.md](CONTRIBUTING.md))
2. Load the data from [data/multi_level_data/](data/multi_level_data/)
3. Create your own analysis notebook
4. Submit a pull request with your findings

### 👨‍🎓 For Students

**Learning from this project:**
- Read the [thesis PDF](legacy/5.%20Visualization%20and%20Reporting/) for research context
- Explore [analysis notebooks](analysis/) to see real research code
- Check [legacy/](legacy/) folder to understand the full research pipeline
- Use [USAGE_GUIDE.md](USAGE_GUIDE.md) to run the code yourself

---

## File Organization

### Essential Files

| File | Purpose | When to Use |
|------|---------|-------------|
| [README.md](README.md) | Project overview and quick start | First time visiting the repo |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Detailed usage instructions | Setting up and running analyses |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute | Want to add to the project |
| [requirements.txt](requirements.txt) | Python dependencies | Installing packages |

### Key Directories

| Directory | Contents | Purpose |
|-----------|----------|---------|
| [analysis/](analysis/) | Jupyter notebooks and R scripts | **Main analysis code - START HERE** |
| [data/multi_level_data/](data/multi_level_data/) | Final processed datasets | The data you'll actually use |
| [legacy/](legacy/) | Original thesis code (2023) | Historical reference, not for reproduction |
| [docs/](docs/) | Documentation | Additional reference materials |
| [output/](output/) | Generated figures and tables | Where analysis results are saved |

---

## Quick Reference: What Goes Where

### Working with Data

**Need data?**
→ [data/multi_level_data/](data/multi_level_data/)

**Documentation about data?**
→ [data/multi_level_data/README.md](data/multi_level_data/README.md)

**Want to understand variables?**
→ [data/multi_level_data/README.md](data/multi_level_data/README.md) (see "Key Variables")

### Running Analysis

**Main exploratory analysis:**
→ [analysis/01_Exploratory_Prominence_Analysis.ipynb](analysis/01_Exploratory_Prominence_Analysis.ipynb) ⭐

**Statistical models (Python):**
→ [analysis/02_Statistical_Models.ipynb](analysis/02_Statistical_Models.ipynb)

**Advanced models (R):**
→ [analysis/03_Multilevel_Models.Rmd](analysis/03_Multilevel_Models.Rmd)

**Instructions for running:**
→ [analysis/README.md](analysis/README.md)

### Understanding Methodology

**Full thesis document:**
→ [legacy/5. Visualization and Reporting/Thesis_UvA_Kaleb_Mazurek.pdf](legacy/5.%20Visualization%20and%20Reporting/)

**Technical details:**
→ [legacy/5. Visualization and Reporting/Technical Report MA Thesis.pdf](legacy/5.%20Visualization%20and%20Reporting/)

**Original pipeline code:**
→ [legacy/](legacy/) (numbered directories 1-5)

---

## Common Tasks

### "I want to reproduce the thesis findings"

1. Install dependencies: `pip install -r requirements.txt`
2. Open main notebook: `jupyter notebook analysis/01_Exploratory_Prominence_Analysis.ipynb`
3. Run cells sequentially
4. Check [analysis/README.md](analysis/README.md) for additional analyses

### "I want to use this data for my own research"

1. Load data from [data/multi_level_data/](data/multi_level_data/)
2. Read [data/multi_level_data/README.md](data/multi_level_data/README.md) for variable descriptions
3. Create your own analysis notebook
4. Cite the dataset (see [README.md](README.md) citation section)

### "I want to understand the research methodology"

1. Read [README.md](README.md) for overview
2. Read thesis PDF in [legacy/5. Visualization and Reporting/](legacy/5.%20Visualization%20and%20Reporting/)
3. Explore original code in [legacy/](legacy/) folders
4. Check technical report for implementation details

### "I want to contribute to this project"

1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Fork the repository
3. Make your changes
4. Submit a pull request

### "I'm getting errors"

1. Check [USAGE_GUIDE.md](USAGE_GUIDE.md) troubleshooting section
2. Verify your environment setup
3. Search [GitHub issues](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)
4. Open a new issue if problem persists

---

## Analysis Workflow

### Recommended Order

```
1. 📖 Read README.md
   ↓
2. 🔧 Follow USAGE_GUIDE.md to set up
   ↓
3. 📊 Run 01_Exploratory_Prominence_Analysis.ipynb
   ↓
4. 📈 Run 02_Statistical_Models.ipynb
   ↓
5. 📐 Run 03_Multilevel_Models.Rmd (if using R)
   ↓
6. 🎓 Read thesis PDF for interpretation
```

### For Quick Exploration

```
1. Install packages: pip install -r requirements.txt
   ↓
2. Open Jupyter: jupyter notebook
   ↓
3. Navigate to: analysis/01_Exploratory_Prominence_Analysis.ipynb
   ↓
4. Run cells and explore!
```

---

## File Naming Conventions

### Analysis Files

- `01_`, `02_`, `03_` - Run in this order
- `_FINAL` suffix - Final version used in thesis
- `.ipynb` - Jupyter notebook (Python)
- `.Rmd` - R Markdown (R code and documentation)

### Data Files

- `FINAL.csv` - Cleaned, validated, analysis-ready
- `multi_level_data.csv` - Complete hierarchical structure
- `level1_FINAL.csv` - Level 1 (mention-level) data only

---

## Repository Philosophy

### What This Repository Provides

✅ **Complete reproducibility** from the processed dataset
✅ **Full transparency** about methods and decisions
✅ **Analysis-ready data** for researchers
✅ **Documented limitations** and known issues

### What It Doesn't Provide

❌ **Full pipeline from raw data** (manual steps involved)
❌ **Raw Congressional Record files** (too large)
❌ **Automated data collection** (requires adaptation)

### Why This Approach?

This repository prioritizes **honest reproducibility** over false claims of complete automation. The `multi_level_data` dataset provides a validated starting point for all thesis findings while preserving the original code for transparency.

---

## Getting Help

### Documentation Hierarchy

1. **Quick questions** → [USAGE_GUIDE.md](USAGE_GUIDE.md) troubleshooting
2. **Setup issues** → [USAGE_GUIDE.md](USAGE_GUIDE.md) installation section
3. **Data questions** → [data/multi_level_data/README.md](data/multi_level_data/README.md)
4. **Analysis questions** → [analysis/README.md](analysis/README.md)
5. **Methodology questions** → Thesis PDF in [legacy/5. Visualization and Reporting/](legacy/5.%20Visualization%20and%20Reporting/)

### Still Stuck?

- Search [GitHub issues](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)
- Open a new issue with details
- Email: kalebmazurek@gmail.com

---

## Version History

- **v2.0.0** (Nov 2024) - Repository modernization, added USAGE_GUIDE and CONTRIBUTING
- **v1.0.0** (Jun 2023) - Original thesis completion

---

## Credits

**Author**: Kaleb Mazurek
**Institution**: University of Amsterdam
**Year**: 2023
**License**: MIT

---

## Quick Links

### Documentation
- [Main README](README.md)
- [Usage Guide](USAGE_GUIDE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Analysis Guide](analysis/README.md)
- [Data Documentation](data/multi_level_data/README.md)

### Analysis Files
- [Main Analysis](analysis/01_Exploratory_Prominence_Analysis.ipynb)
- [Statistical Models](analysis/02_Statistical_Models.ipynb)
- [R Models](analysis/03_Multilevel_Models.Rmd)

### Thesis Documents
- [Full Thesis PDF](legacy/5.%20Visualization%20and%20Reporting/Thesis_UvA_Kaleb_Mazurek.pdf)
- [Technical Report](legacy/5.%20Visualization%20and%20Reporting/Technical%20Report%20MA%20Thesis.pdf)

---

**Last Updated**: November 26, 2024

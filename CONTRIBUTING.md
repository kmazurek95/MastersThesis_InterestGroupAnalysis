# Contributing to Interest Group Prominence Analysis

Thank you for your interest in contributing to this project! This guide will help you get started.

---

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Types of Contributions](#types-of-contributions)
3. [Getting Started](#getting-started)
4. [Development Workflow](#development-workflow)
5. [Code Standards](#code-standards)
6. [Submitting Changes](#submitting-changes)
7. [Community Guidelines](#community-guidelines)

---

## How to Contribute

There are many ways to contribute to this project:

- **Extend the analysis** with new hypotheses or models
- **Improve documentation** (fix typos, clarify instructions, add examples)
- **Report bugs** or data issues
- **Suggest enhancements** to the analysis pipeline
- **Share your research** that builds on this work

All contributions are welcome, from fixing a typo to adding a major new feature!

---

## Types of Contributions

### 1. Analysis Extensions

Add new analyses or visualizations to explore the data:

**Examples:**
- Test alternative statistical models
- Create new visualizations
- Analyze different time periods or subsets
- Explore additional variables

**Where to add:**
- Create a new notebook in `analysis/`
- Name it descriptively (e.g., `04_Alternative_Model_Specifications.ipynb`)
- Document your approach clearly

### 2. Documentation Improvements

Help make this project more accessible:

**Examples:**
- Fix typos or unclear explanations
- Add missing variable definitions
- Expand troubleshooting guides
- Translate documentation

**Files to edit:**
- [README.md](README.md) - Main overview
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed instructions
- [data/multi_level_data/README.md](data/multi_level_data/README.md) - Dataset docs
- [analysis/README.md](analysis/README.md) - Analysis guide

### 3. Bug Reports

Found a problem? Let us know!

**Good bug reports include:**
- Clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, package versions)
- Error messages (full output)

**How to report:**
- [Open a GitHub issue](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)
- Use the "Bug Report" template
- Add relevant labels

### 4. Code Improvements

Enhance the analysis code:

**Examples:**
- Optimize performance
- Add error handling
- Improve code clarity
- Add unit tests

**Guidelines:**
- Follow existing code style
- Add comments for complex logic
- Test your changes
- Update documentation

### 5. Dataset Enhancements

Improve or extend the dataset:

**Examples:**
- Add missing metadata
- Correct data errors
- Validate existing variables
- Document data quality issues

**Important:**
- Clearly document all changes
- Preserve original data
- Explain your methodology

---

## Getting Started

### Prerequisites

1. **GitHub Account**: [Sign up](https://github.com/join) if you don't have one
2. **Git Installed**: [Download Git](https://git-scm.com/downloads)
3. **Development Environment**: Python 3.8+ and/or R 4.0+

### Fork the Repository

1. Go to the [repository page](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis)
2. Click the "Fork" button in the top-right corner
3. This creates your own copy of the repository

### Clone Your Fork

```bash
# Clone your fork (replace YOUR-USERNAME)
git clone https://github.com/YOUR-USERNAME/MastersThesis_InterestGroupAnalysis.git
cd MastersThesis_InterestGroupAnalysis

# Add the original repository as "upstream"
git remote add upstream https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis.git
```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install black flake8 pytest
```

---

## Development Workflow

### 1. Create a Branch

Always create a new branch for your changes:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b descriptive-branch-name

# Examples:
# git checkout -b fix-typo-in-readme
# git checkout -b add-temporal-analysis
# git checkout -b improve-data-loading
```

### 2. Make Your Changes

Edit files, add analyses, or fix bugs:

```bash
# Check what files you've changed
git status

# View your changes
git diff
```

**Best practices:**
- Make small, focused commits
- Test your changes before committing
- Write clear commit messages

### 3. Commit Your Changes

```bash
# Stage your changes
git add file1.py file2.md

# Or stage all changes
git add .

# Commit with a descriptive message
git commit -m "Add temporal trend analysis to notebook 04"
```

**Good commit messages:**
- Use present tense ("Add feature" not "Added feature")
- Be specific and concise
- Reference issues when relevant ("Fix #42")

**Examples:**
```bash
git commit -m "Fix typo in USAGE_GUIDE.md"
git commit -m "Add coefficient plot to statistical models notebook"
git commit -m "Update dataset README with missing variable descriptions"
```

### 4. Push to Your Fork

```bash
# Push your branch to your fork
git push origin descriptive-branch-name
```

### 5. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Select your branch
4. Write a clear description:
   - What changes did you make?
   - Why did you make them?
   - How did you test them?
5. Submit the pull request

**Pull request title examples:**
- `Add alternative model specifications notebook`
- `Fix data loading error in main analysis`
- `Update USAGE_GUIDE with R troubleshooting`

---

## Code Standards

### Python Style

Follow [PEP 8](https://pep8.org/) style guidelines:

```python
# Good
def calculate_prominence_rate(df):
    """Calculate the proportion of prominent mentions.

    Args:
        df (pd.DataFrame): DataFrame with 'prominent' column

    Returns:
        float: Proportion of prominent mentions
    """
    return df['prominent'].mean()

# Use descriptive variable names
prominence_rate = calculate_prominence_rate(data)
```

**Tools:**
```bash
# Format code automatically
black your_script.py

# Check for style issues
flake8 your_script.py
```

### R Style

Follow [tidyverse style guide](https://style.tidyverse.org/):

```r
# Good
calculate_prominence_rate <- function(df) {
  # Calculate proportion of prominent mentions
  df %>%
    summarize(prominence_rate = mean(prominent, na.rm = TRUE)) %>%
    pull(prominence_rate)
}

# Use descriptive names and consistent indentation
prominence_rate <- calculate_prominence_rate(data)
```

### Jupyter Notebooks

**Structure:**
1. **Title cell** (Markdown): Clear title and description
2. **Setup cell**: Import libraries
3. **Analysis cells**: Well-documented code
4. **Results cells**: Clear outputs and visualizations

**Best practices:**
- Clear markdown headers for sections
- Comments explaining complex code
- Meaningful variable names
- Remove unnecessary cells before committing

**Example:**
```python
# Cell 1: Markdown
# # Temporal Analysis of Interest Group Prominence
# This notebook analyzes trends in prominence over time.

# Cell 2: Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 3: Load data
data = pd.read_csv('data/multi_level_data/df_interest_group_prominence_FINAL.csv')

# Cell 4: Analysis with clear comments
# Calculate prominence by year
yearly_prominence = (
    data.groupby('year')['prominent']
    .mean()
    .reset_index()
)
```

### Documentation

**Use clear, concise language:**
- Write for your audience (researchers, students, practitioners)
- Explain jargon and technical terms
- Provide examples
- Use proper grammar and spelling

**Markdown formatting:**
```markdown
# Headers for sections
## Subsections
### Sub-subsections

**Bold** for emphasis
*Italic* for terms
`code` for inline code

- Bullet lists
1. Numbered lists

[Links](https://example.com)
```

---

## Submitting Changes

### Pull Request Checklist

Before submitting, ensure:

- [ ] Code runs without errors
- [ ] New code is documented
- [ ] Relevant documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] Tests pass (if applicable)

### Review Process

1. **Automated checks**: GitHub Actions may run tests
2. **Code review**: Maintainer will review your changes
3. **Feedback**: You may be asked to make revisions
4. **Merge**: Once approved, your changes will be merged!

**Response time:**
- We aim to review PRs within 1-2 weeks
- Complex changes may take longer
- Feel free to ping if no response after 2 weeks

---

## Community Guidelines

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

**In brief:**
- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community

### Getting Help

**Questions?**
- Check [USAGE_GUIDE.md](USAGE_GUIDE.md) first
- Search [existing issues](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)
- Open a new issue with your question
- Email: kalebmazurek@gmail.com

**Stuck on something?**
- Include your environment details
- Share the error message
- Describe what you've tried
- Provide a minimal example

---

## Recognition

Contributors will be acknowledged in:
- Repository README
- Release notes
- Academic citations (for substantial contributions)

Thank you for contributing to open science!

---

## Additional Resources

### Learning Resources

**Git and GitHub:**
- [GitHub Guides](https://guides.github.com/)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [First Contributions](https://github.com/firstcontributions/first-contributions)

**Python:**
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

**R:**
- [R for Data Science](https://r4ds.had.co.nz/)
- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)

**Multilevel Modeling:**
- [GLMM FAQ](https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html)
- [lme4 Package](https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

Feel free to reach out:
- **Email**: kalebmazurek@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis/discussions)

---

**Thank you for contributing to research transparency and open science!**

---

**Last Updated**: November 26, 2024

# GitHub Upload Checklist

**Complete this checklist before pushing to GitHub**

---

## Pre-Upload Verification

### 📁 Files and Structure

- [x] All analysis notebooks renamed and organized
  - [x] `01_Exploratory_Prominence_Analysis.ipynb` (main analysis)
  - [x] `02_Statistical_Models.ipynb`
  - [x] `03_Multilevel_Models.Rmd`

- [x] Documentation files created
  - [x] `README.md` (updated with current structure)
  - [x] `USAGE_GUIDE.md` (comprehensive instructions)
  - [x] `CONTRIBUTING.md` (contribution guidelines)
  - [x] `REPOSITORY_GUIDE.md` (navigation guide)
  - [x] `requirements.txt` (Python dependencies)

- [x] Configuration files
  - [x] `.gitignore` (comprehensive)
  - [x] `.gitattributes` (Git LFS for large files)
  - [x] `CITATION.cff` (citation metadata)
  - [x] `LICENSE` (MIT license)

### 📊 Data Files

- [ ] Verify data files are present in `data/multi_level_data/`:
  - [ ] `df_interest_group_prominence_FINAL.csv`
  - [ ] `level1_FINAL.csv`
  - [ ] `multi_level_data.csv`
  - [ ] `README.md` (dataset documentation)

**⚠️ IMPORTANT**: These CSV files are large (>100MB each)

**Options for handling large files:**

1. **Git LFS (Recommended)**
   ```bash
   # Install Git LFS
   git lfs install

   # Track large CSV files
   git lfs track "*.csv"

   # Commit and push
   git add .gitattributes
   git add data/multi_level_data/*.csv
   git commit -m "Add dataset files via Git LFS"
   git push
   ```

2. **External Hosting**
   - Host files on institutional server, Zenodo, or OSF
   - Add download links to [data/multi_level_data/README.md](data/multi_level_data/README.md)
   - Update main README with access instructions

3. **Release Attachments**
   - Create a GitHub release
   - Attach CSV files to the release
   - Update README with download links

### 📝 Content Review

- [ ] Review all markdown files for:
  - [ ] Broken links
  - [ ] Typos
  - [ ] Correct file paths
  - [ ] Up-to-date information

- [ ] Review notebooks:
  - [ ] Clear all output cells (optional, for cleaner diffs)
  - [ ] Ensure no sensitive data or API keys
  - [ ] Check that file paths are relative (not absolute)

### 🔒 Security Check

- [ ] No API keys or credentials in code
- [ ] No absolute paths with personal info (e.g., `C:\Users\YourName\`)
- [ ] No sensitive data in notebooks
- [ ] `.env` files are in `.gitignore`

### 🎨 Repository Presentation

- [ ] Add repository description on GitHub:
  ```
  Master's thesis analyzing advocacy group prominence in U.S. Congressional debates using NLP, machine learning, and multilevel modeling (2014-2018)
  ```

- [ ] Add topics/tags:
  - `political-science`
  - `congressional-record`
  - `interest-groups`
  - `machine-learning`
  - `natural-language-processing`
  - `multilevel-modeling`
  - `research`
  - `thesis`
  - `python`
  - `r`

- [ ] Set repository settings:
  - [ ] Enable Issues
  - [ ] Enable Discussions (optional)
  - [ ] Add repository website (if applicable)

---

## Upload Commands

### Initial Upload (First Time)

```bash
# 1. Verify you're in the repository root
pwd
# Should show: .../MastersThesis_InterestGroupAnalysis

# 2. Check current status
git status

# 3. Review changes
git diff

# 4. Stage all new files
git add .

# 5. Create commit
git commit -m "Prepare repository for GitHub upload

- Organize analysis notebooks with clear numbering
- Add comprehensive documentation (USAGE_GUIDE, CONTRIBUTING, REPOSITORY_GUIDE)
- Update README with current structure
- Add requirements.txt for Python dependencies
- Configure .gitattributes for large files
- Update analysis README with new structure"

# 6. Verify remote is set
git remote -v
# Should show your GitHub repository

# 7. Push to GitHub
git push -u origin main
```

### If You Need to Set Up Git LFS First

```bash
# Install Git LFS (one-time setup)
git lfs install

# Track large CSV files
git lfs track "*.csv"
git lfs track "*.parquet"

# Add .gitattributes
git add .gitattributes

# Commit LFS configuration
git commit -m "Configure Git LFS for large data files"

# Add data files
git add data/multi_level_data/*.csv

# Commit data
git commit -m "Add dataset files via Git LFS"

# Push everything
git push -u origin main
```

---

## Post-Upload Verification

### On GitHub Website

- [ ] Verify all files are uploaded correctly
- [ ] Check that README displays properly
- [ ] Test navigation links in README
- [ ] Verify notebooks render correctly
- [ ] Check that data files are accessible

### Test Clone and Setup

```bash
# Clone in a new directory to test
cd /tmp
git clone https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis.git
cd MastersThesis_InterestGroupAnalysis

# Test setup process
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify data is accessible
ls data/multi_level_data/

# Test opening a notebook
jupyter notebook analysis/01_Exploratory_Prominence_Analysis.ipynb
```

### Create Release (Optional)

```bash
# Create a tag
git tag -a v2.0.0 -m "Version 2.0.0: Repository modernization with comprehensive documentation"

# Push the tag
git push origin v2.0.0
```

Then on GitHub:
1. Go to Releases
2. Draft a new release
3. Choose the tag (v2.0.0)
4. Title: "Version 2.0.0: Repository Modernization"
5. Description: Summarize changes
6. Attach large data files if not using Git LFS
7. Publish release

---

## Documentation Links to Update

After uploading, update any external references:

- [ ] Update your CV/resume with repository link
- [ ] Update your personal website
- [ ] Update your LinkedIn profile
- [ ] Share on academic social media (ResearchGate, Academia.edu)
- [ ] Add to your Google Scholar profile

---

## Sharing and Promotion

### Social Media Posts

**LinkedIn/Twitter template:**
```
Excited to share my Master's thesis repository! 🎓

Analyzing why some advocacy groups get more recognition from politicians than others, using:
📊 77,000+ Congressional Record documents (2014-2018)
🤖 Machine learning (81% accuracy)
📈 Multilevel statistical modeling

Fully reproducible with comprehensive documentation.

#OpenScience #PoliticalScience #DataScience #Research

🔗 https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis
```

### Email to Advisors/Committee

Subject: Thesis Repository Now Available on GitHub

Body:
```
Dear [Name],

I'm pleased to share that I've made my thesis research fully available on GitHub with comprehensive documentation and reproducible analysis:

https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis

The repository includes:
- All analysis code and data
- Step-by-step usage guide
- Original thesis PDF and technical report
- Contribution guidelines for others to build on this work

Thank you again for your guidance throughout this project.

Best regards,
Kaleb Mazurek
```

---

## Maintenance Plan

### Regular Updates

- [ ] Monitor GitHub Issues weekly
- [ ] Respond to questions and problems
- [ ] Review and merge pull requests
- [ ] Update documentation as needed
- [ ] Fix broken links periodically

### Future Enhancements

Consider adding:
- [ ] Automated testing (GitHub Actions)
- [ ] Jupyter Book for documentation
- [ ] Interactive visualizations (Plotly Dash)
- [ ] Docker container for reproducibility
- [ ] Binder/Colab links for easy exploration

---

## Troubleshooting

### Large Files Won't Upload

**Error**: File larger than 100 MB

**Solutions**:
1. Use Git LFS (recommended)
2. Split files into smaller chunks
3. Host files externally (Zenodo, OSF, institutional server)
4. Use release attachments

### Push Rejected

**Error**: `! [rejected] main -> main (fetch first)`

**Solution**:
```bash
git pull origin main --rebase
git push origin main
```

### .gitignore Not Working

**Problem**: Files in .gitignore are still being tracked

**Solution**:
```bash
# Remove from tracking but keep locally
git rm --cached filename

# Or for a directory
git rm -r --cached directory/

# Commit the change
git commit -m "Remove ignored files from tracking"
```

---

## Final Checklist

Before making the repository public:

- [ ] All documentation complete
- [ ] All sensitive data removed
- [ ] Data files handled properly (LFS or external)
- [ ] Links tested
- [ ] Repository settings configured
- [ ] First push successful
- [ ] Clone test completed
- [ ] README displays correctly
- [ ] Ready to share!

---

## Support

If you encounter issues during upload:

- [GitHub Docs: Adding Files](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [Git LFS Documentation](https://git-lfs.github.com/)
- Email: kalebmazurek@gmail.com

---

**Good luck with your GitHub upload! 🚀**

---

**Last Updated**: November 26, 2024

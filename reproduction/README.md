# Reproduction

A self-contained check that this repo's regression layer reproduces from the pinned, on-disk data. Run in July 2026 as part of a reproducibility audit of the thesis pipeline.

## What it shows

The multilevel logistic regression (the thesis's inferential model) reproduces exactly from the archived integrated dataset `data/multi_level_data/level1_FINAL.csv`:

- Python fixed-effects logistic (`reproduce_regression.py`) re-runs the 9 GLM fits from `analysis/02_Statistical_Models.ipynb` and matches the saved `analysis/output/model_parameters.csv` on all 50 coefficients to machine precision (max |Δ| ≈ 9e-17).
- R crossed random-effects (`_run_glmer.R`) re-runs the `lme4::glmer` models from `analysis/03_Multilevel_Models.Rmd` and matches the rendered `.md` to printed precision (empty-model Var(org)=1.23, Var(issue)=0.29, N=13,417, ICC_org≈0.26).

The classifier labels are the one stage not reproducible from scratch: the 2023 deployed re-fit was unseeded and its code was not saved. They are pinned in `level1_prominence` inside `level1_FINAL.csv`, so everything downstream reproduces.

## Scope and honesty note

This reproduces the reorg's own saved outputs exactly, and it is the same model family as the submitted thesis (multilevel logistic, crossed organization and policy-area random effects, odds ratios) with the same substantive conclusions. It is not a numeric match to the thesis PDF's reported salience odds ratios. The thesis's Model 1 reports medium OR 1.489 and high OR 0.702 (both non-significant), while this re-analysis runs on a somewhat different sample. See `REGRESSION_REPRODUCTION.md` for the full comparison. The takeaway that survives every specification: medium-salience robustly predicts prominence; the high-salience effect is not robust.

## Files

- `run_pipeline.sh`: one runner that verifies the pinned dataset's sha256, then runs both stages and self-checks.
- `reproduce_regression.py`: Python anchor (statsmodels logistic) compared to the saved parameters.
- `_run_glmer.R`: R anchor (`lme4::glmer` crossed random effects) compared to the rendered results.
- `REGRESSION_REPRODUCTION.md`: the full write-up covering both anchors, the PDF cross-check, and the scaling seam.
- `PHASE0_VERIFICATION.md`: byte-level (sha256) verification of the data files the run used.

## Run it

```
git lfs pull                 # hydrate data/multi_level_data/*.csv
cd reproduction
bash run_pipeline.sh         # set RSCRIPT=/path/to/Rscript if Rscript is not on PATH
```

Needs Python (numpy, pandas, statsmodels) and R (dplyr, lme4, broom.mixed). All paths are derived from each script's own location, so it runs from any clone.

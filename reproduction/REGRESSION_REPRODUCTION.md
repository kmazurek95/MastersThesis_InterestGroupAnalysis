# REGRESSION_REPRODUCTION.md: Phase 1 (the anchor)

Question: does the thesis/reorg regression reproduce from on-disk data only?

Answer: yes, exactly, on both the Python (fixed-effects logistic) and the R (`lme4::glmer` crossed random-effects) implementations.

Generated 2026-07-19 (PDF cross-check corrected 2026-07-20). All inputs read-only; no data downloaded. Run from `reproduction/` inside the repo (`bash run_pipeline.sh`).

---

## What the "regression" is (and a clarification)

The reorg (`MastersThesis_InterestGroupAnalysis`) models prominence as a per-mention binary logistic outcome, with 3 model sets (A: issue salience, B: politician characteristics, C: organizational resources), each an empty model plus two nested models. Two parallel implementations exist:

- Python `analysis/02_Statistical_Models.ipynb` uses a `statsmodels` fixed-effects logistic GLM (`smf.glm(..., family=Binomial())`). It produced the saved `analysis/output/model_parameters.csv`.
- R `analysis/03_Multilevel_Models.Rmd` uses true crossed random effects, `lme4::glmer(prominence ~ ... + (1|org_id) + (1|issue_area), family=binomial, nAGQ=0, bobyqa)`. It produced the rendered `analysis/03_Multilevel_Models.md`.

> PDF-verified headline model (checked 2026-07-20). The submitted thesis `docs/Thesis_UvA_Kaleb_Mazurek.pdf`
> uses a mixed-effects (crossed random-effects) logistic regression: binary prominence DV, odds ratios,
> Org and Policy-Area random effects, Models 1/2/3 (salience/politician/org), each empty+IV+full.
> Verbatim: *"a Generalized Linear Mixed-Effects Model … the binary nature of the dependent variable"*;
> *"Summary of mixed-effects logistic regression models"*; interpreted in *"odds ratios"* (71 mentions).
> The PDF has zero mentions of negative-binomial, Poisson, IRR, or overdispersion.
> So the logistic reproduction is the thesis headline model. The `glmer.nb` files in
> `UVA_RMSS_THESIS_MAZUREK/Models/*.Rmd` are an alternative/exploratory count lineage that did not
> become the thesis headline (correcting an earlier note that had this backwards).
>
> Coefficient cross-check (corrected 2026-07-20). This reproduction reproduces the reorg's own
> saved outputs exactly (Python 50/50 coefficients to machine precision; R `glmer` to printed precision),
> which is what Phase 1 proves. It is not a numeric match to the 2023 thesis PDF's salience odds ratios.
> Verified against the PDF (pages 20, 26–27, columns de-scrambled with an exp() cross-check): the thesis's
> Model 1 (salience + chamber + party, N≈15,642) reports medium OR 1.489 and high OR 0.702, both
> non-significant (p≈0.07 / 0.08). The reorg re-analysis uses a different sample and salience structure
> (salience-only A1: medium 2.04 / high 1.53, N=13,417; full A2: medium 1.42 / high 1.06, N=7,030). So it is
> the same model family with the same qualitative conclusion (medium-salience robustly positive;
> high-salience not robust across specs), but the earlier claim that the reproduction "matches the PDF's ORs
> exactly (2.03/1.41)" was wrong: 2.03/1.41 are exp() of the reorg's own coefficients
> (exp(0.7125)/exp(0.3518)), values the PDF does not contain.
>
> One documented scaling seam: the thesis reports ORs for standardized continuous predictors
> (per-1-SD; e.g. Bills Sponsored OR≈1.82, the `StandardScaler` step is in
> `legacy/4/DataProcessingAndRegression.py`), whereas the reproduced models use raw units (per-1-unit;
> OR≈1.00–1.05). Standardization rescales only the continuous vars' own odds ratios; categoricals, fit, and
> significance are unchanged.

## Inputs (all on disk, pinned)

- `MastersThesis_InterestGroupAnalysis/data/multi_level_data/level1_FINAL.csv` (107 MB, 22,414 rows × 175 cols; sha256 `c8781d5a…`)
- Reference outputs: `analysis/output/model_parameters.csv` (Python) and `analysis/03_Multilevel_Models.md` (R)
- Runner scripts (in `reproduction/`): `reproduce_regression.py` (Python), `_run_glmer.R` (R, reads `level1_FINAL.csv` directly), `run_pipeline.sh` (orchestrator)

## Anchor 1: Python fixed-effects logistic vs `model_parameters.csv`

Re-ran the notebook's exact preprocessing plus 9 GLM logistic fits on `level1_FINAL.csv`, joined to the saved parameters on (model, term):

| metric | max &#124;Δ&#124; (saved minus reproduced) |
|---|---|
| estimate | 9.02e-17 |
| std_error | 9.71e-17 |
| z_value | 8.88e-16 |
| p_value | 9.95e-17 |
| odds_ratio | 2.22e-16 |

All 50 coefficients matched, 0 unmatched. Verdict: exact / byte-level (differences are floating-point round-off). Data: 22,414 to 22,411 rows after dropping "(1) Corporations"; prominence base rate 46.69%.

## Anchor 2: R `lme4::glmer` crossed RE vs `03_Multilevel_Models.md`

`nAGQ=0` plus bobyqa is deterministic (no seed dependence). Reproduced values vs the rendered `.md`:

| model | quantity | reference (.md) | reproduced |
|---|---|---|---|
| Empty | Var(org_id) | 1.23 | 1.226 |
| Empty | Var(issue_area) | 0.29 | 0.290 |
| Empty | N obs | 13,417 | 13,417 |
| Empty | ICC(org) / ICC(issue) | n/a | 0.255 / 0.060 |
| A1 | saliency medium (z) | 0.712 (4.83) | 0.7125 (4.832) |
| A1 | saliency high (z) | 0.428 (2.83) | 0.4282 (2.830) |
| A2 | saliency medium (SE) | 0.3518 (0.1414) | 0.3518 (0.1414, z 2.488) |

Verdict: exact (to printed precision). Model Ns reproduced: A1 13,417 · A2 7,030 · B1/B2 6,945 · C1 13,034 · C2 6,896.

### Why N differs across models (not an error)

- Python fixed-effects models keep all rows with non-missing predictors (empty N ≈ 22,411).
- R `glmer` models include `(1|issue_area)` as a grouping factor, so rows with missing `issue_area` are dropped, giving an empty-model N = 13,417 (≈6,600 rows lack an issue area). This is inherent to the random-effects specification, and it reproduces exactly.

## Key substantive results (reproduced)

- Salience is non-monotonic: medium-salience areas most strongly predict prominence; the high-salience effect attenuates once controls enter (A2 high = 0.06, n.s.).
- The seniority reversal (Model B) and the external-lobbyist effect (Model C, p≈0.001) both reproduce.
- Random-effect structure: substantial org-level clustering (ICC ≈ 0.26), modest issue-area clustering (≈0.06).

## Bottom line

The regression layer is fully reproducible from on-disk data: Python to machine precision, R `glmer` to printed precision. Nothing about the regression is a reproduction risk. The only non-deterministic artifact anywhere in the pipeline is upstream (the classifier labels), addressed in Phase 2.

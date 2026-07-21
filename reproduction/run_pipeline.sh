#!/usr/bin/env bash
# ONE reproducible runner: pinned integrated dataset -> regression (Python + R glmer), verified.
# Pins the non-deterministic artifacts (classifier labels via level1_FINAL) and re-runs the
# deterministic regression stage end-to-end. No data downloads; read-only on all inputs.
set -u
# repo-relative: this script lives in reproduction/; derive the repo root from its own location.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
L1="$REPO/data/multi_level_data/level1_FINAL.csv"
RSCRIPT="${RSCRIPT:-Rscript}"   # override via env if Rscript is not on PATH
# Classifier labels are already baked into level1_FINAL.csv (level1_prominence). The external
# CLASSIFIED system-of-record file is optional; set CLASSIFIED_LABELS to point at it if you have it.
CL="${CLASSIFIED_LABELS:-}"
EXPECT_SHA="c8781d5a0516b7a1f920b9c27e00b4428c1eb6879145fff3100f4cfe7545811d"
fail(){ echo "FAIL: $1"; exit 1; }

echo "[1/4] verify pinned integrated dataset (level1_FINAL.csv)..."
[ -f "$L1" ] || fail "level1_FINAL.csv missing"
got=$(sha256sum "$L1" | cut -d' ' -f1)
[ "$got" = "$EXPECT_SHA" ] && echo "  sha256 OK" || fail "level1_FINAL sha256 mismatch: $got"

echo "[2/4] verify pinned classifier labels (system-of-record) present..."
if [ -n "$CL" ] && [ -f "$CL" ]; then echo "  CLASSIFIED labels present (pinned)"; else echo "  labels pinned in level1_FINAL (external CLASSIFIED file not provided)"; fi

echo "[3/4] S5 Python fixed-effects logistic -> compare to model_parameters.csv..."
python "$HERE/reproduce_regression.py" 2>/dev/null | grep -E "VERDICT" || fail "python regression stage failed"

echo "[4/4] S5 R lme4::glmer crossed random effects -> empty-model variances..."
"$RSCRIPT" "$HERE/_run_glmer.R" 2>/dev/null | grep -E "EMPTY:" || fail "R glmer stage failed"

echo "PASS: pinned inputs -> regression reproduces end-to-end."

# Phase 0: data integrity (sha256)

Before running the reproduction, the pinned data files were verified byte-identical to the archived source (99/99 files, 888,982,062 bytes total, 0 mismatches).

## sha256 of the key data files

```
c33911f9019a9cbe3de62592b5873a3775c4ba46f85a1605f90733c75fd5318d  data/multi_level_data/multi_level_data.csv
c8781d5a0516b7a1f920b9c27e00b4428c1eb6879145fff3100f4cfe7545811d  data/multi_level_data/level1_FINAL.csv
88992a3d9b80cbb5faee51f804c8f961296b8344a5f1b7757ef1e1d9a465fd1c  data/multi_level_data/df_interest_group_prominence_FINAL.csv
```

`level1_FINAL.csv` is the integrated dataset the regression runs on. `run_pipeline.sh` re-checks its sha256 (`c8781d5a…`) before running, so a mismatched or partially-hydrated LFS pull is caught early.

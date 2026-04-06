# Release Assets

This directory contains lightweight public assets that are directly relevant to reproducing the manuscript results and can be redistributed safely.

## Included Files

- `agiqa3k_split_seed42.json`
  Fixed AGIQA-3K split file used by the main rerun reported in the manuscript.
- `logic_feature_names.json`
  Canonical order of the 12-dimensional logic interface consumed by the online scorer.
- `logic_cache_schema.md`
  Field-level description of the Stage A cache and the compressed 12-dimensional logic vector.

## Intended Use

These files support:

- fixed-split reruns on AGIQA-3K
- verification that the released 12-dimensional interface matches the manuscript definition
- inspection of the cache fields used between Stage A auditing and online residual scoring

## Redistribution Note

This repository does not redistribute the original AGIQA-3K or AIGCIQA2023 images and annotations. Derived artifacts that depend on benchmark redistribution restrictions should be archived separately or shared according to the manuscript's data-availability statement.

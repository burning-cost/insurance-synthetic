# Changelog

## [0.1.6] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.1.4 (2026-03-22) [unreleased]
- refactor: convert benchmark to Databricks notebook format
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.4 (2026-03-21)
- docs: replace pip install with uv add in README
- fix: exclude severity from vine copula; fit conditional marginal on claimers only
- Add community CTA to README
- Add MIT license
- Add benchmark results to Performance section with actual Databricks run data
- QA batch 10: fix frequency architecture, fallback warning, TSTR loss, dead code
- refresh benchmark numbers post-P0 fixes
- fix: P0/P1 bugs in fidelity, marginals, and synthesiser (v0.1.3)
- Add standalone benchmark: VineCopulaSynthesizer vs independent shuffle
- Add benchmark: vine copula vs naive independent sampling
- fix: expand expected distribution families in test_marginals assertions
- Remove redundant Capabilities section (covered by Performance)
- Add ## Performance section to README
- docs: add Databricks notebook link
- fix: move pyvinecopulib to optional vine extra (was accidentally required in 0.1.0)
- Add Related Libraries section to README
- fix: add seed data source pointer and inline fallback to quick start


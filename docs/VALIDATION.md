# Validation

Validated on October 15, 2026 with Python 3.10.22 on macOS 15.7.3 (arm64). The pinned scientific/test dependencies are in `requirements-lock.txt`; `environment.yml` adds Jupyter through `requirements-notebooks.txt`.

The documented Conda environment was created successfully with Conda 23.3.1 in an isolated temporary prefix. `python -m pip check` reported no broken requirements, and the Jupyter command was available. The README's test and synthetic-data commands were executed from the repository root in that environment. All 13 tests passed, and the example produced eight effect-estimate rows, four window-panel PDFs, outcome-count and monthly-trajectory PDF/PNG figures, and CSV source tables. The combined test/example walkthrough took approximately 44 seconds on the validation machine, excluding environment installation; this is not a full-study runtime estimate. Documentation links and shell-block syntax were also checked.

All 13 automated tests pass, covering:

- Exactly the four manuscript outcome windows and their boundary assignments.
- Pre-enrollment diagnosis binarization, retained utilization counts, and preserved follow-up claim counts.
- Mean imputation of age/ADI.
- The unregularized nonnegative, sum-to-one balancing objective against an analytically solvable example.
- Rejection of analysis settings outside the final workflow.
- Gaussian intervals, two-sided tests, Bonferroni adjustment, and undefined ratios.
- Patient-token weight alignment and error handling.
- Exact ICD matching and threaded input reading.
- Enforcement of the final outcome-manifest size and negative-control count.
- End-to-end synthetic weighting, bootstrap estimation, and figures with one worker and two spawned workers.

The synthetic workflow uses the same representation, objective, and inference as the study configurations, with CLARABEL in place of licensed MOSEK and fewer artificial outcomes/resamples. The MOSEK package is installed, but a licensed full-data solve was not performed. The retained notebooks pass schema, syntax, and name checks and contain no saved outputs. Full-data numerical reproduction is not asserted; missing inputs and analyses are listed in [AVAILABILITY.md](AVAILABILITY.md).

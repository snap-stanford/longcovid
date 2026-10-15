# Validation

Validated on October 15, 2026 with Python 3.10.22 and the pinned dependencies. The complete resolved environment is in `requirements-lock.txt`.

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

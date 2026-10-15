# Analysis definitions

The main manuscript's Methods M2–M3 and the supplement's Extended Methods S1–S3 define the single implementation included here.

## Cohorts

The three study configurations use verified, externally constructed membership files:

- **Test-based controls:** first PCR test between March 1, 2020 and March 1, 2021; no prior evidence of COVID-19; positive diagnosis within seven days versus no evidence within seven days; survival through day 30; controls aligned to case enrollment dates at a daily 5:1 ratio.
- **Unrestricted controls:** confirmed COVID-19 diagnosis versus no evidence of COVID-19, with control enrollment dates sampled from the case enrollment-date distribution and survival through day 30.
- **Negative-control exposure:** first negative PCR test with no evidence of COVID-19 before or after testing versus unrestricted controls, whose enrollment dates follow the placebo cases' dates; survival through day 30.

The 2019-claim eligibility restriction and the April 4, 2023 observation cutoff belong to the upstream cohort/export step. That step is not implemented by the supplied final materials and is not replaced with a different SQL iteration.

## Pre-enrollment covariates

The model uses 48 consecutive 30-day bins covering days −1440 through −1. Clinical outcomes are restricted to the final 614-outcome manifest and binarized within each pre-enrollment bin. Healthcare-utilization covariates retain visit counts for inpatient, outpatient, home, emergency-room, ambulance, pharmacy, and non-hospital institution visits. Payer-complete month indicators are also included.

Static covariates are gender indicators, age, ADI, mean income, non-white or Hispanic fraction, and fraction without a bachelor's degree. Age and ADI are mean-imputed across the selected analysis cohort. ZIP-3 ADI and Census linkage values must be supplied using the final population-weighted reference tables. Each balancing column is normalized to unit Euclidean norm.

## Control weighting

For control matrix `X_control` and case mean `mean_cases`, solve:

```text
minimize ||X_control.T @ weights - mean_cases||_2
subject to weights >= 0
           sum(weights) = 1
```

All three study designs use this objective and representation.

## Follow-up and inference

Follow-up claim counts are summed within four half-open windows: `[0, 30)`, `[30, 120)`, `[120, 360)`, and `[360, 720)`. The first two raw bins, day zero and days 1–29, contribute additively to the acute window. Pre-enrollment binarization does not alter follow-up counts.

RR is the case mean divided by the weighted-control mean. RD is their difference divided by the number of months in the window: 1, 3, 8, and 12. Patient bootstrap resampling uses 1,000 replicates and holds fitted control weights fixed. Standard errors are the standard deviation of replicate statistics. Confidence intervals are Gaussian and centered on the observed estimate; tests are two-sided. Bonferroni adjustment uses all 614 outcomes within each analysis window. Unadjusted results remain available for negative-control comparisons.

`attributable` requires RR ≥ 1.1 and adjusted p < 0.05. `confident_RR_at_least_1_1` additionally requires the adjusted lower confidence limit ≥ 1.1. Undefined bootstrap ratios retain undefined inference results.

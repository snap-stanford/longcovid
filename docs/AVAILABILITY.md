# Code and data availability

This repository provides monthly feature construction, constrained covariate balancing, effect estimation for the four follow-up windows, bootstrap inference, figure exports, and notebooks for descriptive tables, geographic comparisons, and ICD-code tables. The simulated example exercises the analysis without access to patient records.

## Study data and reference inputs

Patient-level claims and cohort records are not distributed with the code. Running the study-data workflow requires the cohort and claims exports, the 614-outcome manifest, the expanded ICD mapping, and the Census/ADI linkage tables described in [DATA.md](DATA.md). These study inputs must be supplied separately.

The cohort-construction and export implementation, study reference inputs, and archived study outputs are not included. Analysis can begin with verified prepared cohorts in the format described in the data guide, or with the required exports for preprocessing.

## Analyses outside this repository

The enrollment-period, age-category, local-incidence, patient-identification, and geographic/quarter balance sensitivity analyses are not included. The plotting code produces analysis figures and source tables; exact journal panel assembly and artwork are not included.

## Reproducing results

The automated checks and simulated example validate the implemented workflow. Reproducing the paper's numerical results requires the study inputs and comparison with the original study outputs. Full-data numerical reproduction has not been verified. See [VALIDATION.md](VALIDATION.md) for the checks performed.

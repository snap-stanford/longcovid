# Input specifications

Patient-level study data are supplied separately in the authorized workspace. The code release contains no claims, patient-linked weights, or preliminary diagnosis lists.

## Final reference inputs

| Default path under `LONGCOVID_HOME` | Required columns |
| --- | --- |
| `data/reference/analysis_outcomes.csv` | Unique `OUTCOME`, `negative` (true/false); optional `Diagnosis Group`. Exactly 614 rows, including 49 negative controls. |
| `data/raw_data/diagnosis_codes_to_categories_processed.csv` | `Diagnosis`, `Processed Codes ICD10`, `Processed Codes ICD9`; final expanded comma-separated code identifiers. |
| `data/raw_data/dep_index.csv` | `ZIP_3`, `ADI_NATRANK`; final population-weighted ZIP-3 linkage values. |
| `data/census/processed_census_data.csv` | `ZIP3`, `Mean Income`, `white_percentage`, and the education percentage field below. |

The education field is `Estimate!!Total!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher Percentage`. The preprocessing takes its complement. `white_percentage` is the non-Hispanic white fraction; its complement becomes `nonwhite_or_hispanic` in the model. Preserve leading zeros in ZIP-3 and ICD identifiers. Diagnosis identifiers must agree with the manifest after the source label normalization of spaces/slashes to underscores.

Override reference paths using `LONGCOVID_PROCESSED_DIAGNOSES`, `LONGCOVID_ADI`, or `LONGCOVID_CENSUS`. Wildcard expansion, provisional diagnosis-list combinations, and ADI-generation alternatives are not part of this release.

## Claims and cohort exports

Exports are tab-delimited and may be gzip-compressed. Files must be globally sorted by `PATIENT_ID` and retain the source partition suffix `n_n_n`. Each enrollment row requires a unique, contiguous patient ID and the same ID mapping across its claims/static exports. The cohort membership CSV selects the correct enrollment rows and supplies the final case/control labels.

| Directory under `data/raw_data/` | Filename fragment | Used columns |
| --- | --- | --- |
| `cohort_<study>/` | `cohort_.tsv` | `PATIENT_ID`, `PATIENT_TOKEN_2`, `CLAIM_DATE`, `DATE_OF_DEATH`, `TREATMENT` |
| `medical_headers_with_visit_id/` | `medical_headers_with_id_.tsv` | `PATIENT_ID`, `ICD_9`, `DAYS_SINCE_2012`, `DIAGNOSIS_ARRAY` |
| `static_covariates/` | `static_.tsv` | `PATIENT_ID`, `PATIENT_DOB`, `PATIENT_GENDER`, `PATIENT_ZIP3` |
| `filtered_visit_types_with_visit_id/` | `filtered_visit_types_with_id_.tsv` | Quoted source header names for `VISIT_ID_SIMPLE`, `PATIENT_ID`, `START_DAYS_SINCE_2012`, `VISIT_TYPE` |
| `closed_events/` | `closed_events_.tsv` | `PATIENT_ID`, `DAYS_SINCE_2012` |
| `ptf_files/<study>.csv` | CSV | Unique `PATIENT_ID`, binary `TREATMENT` |

The preprocessing uses the named visit categories and original Snowflake null/string encoding. Clinical code arrays must contain complete identifiers matching the supplied processed ICD table. The final cohort/export SQL is not included.

## Prepared cohort alternative

Analysis can start directly from verified prepared cohorts:

```text
cohort_directory/
  csr_matrices/X_chunk_0.npz ...
  csr_matrices/Y_chunk_0.npz ...
  np_arrays/Patient_token.npy
  np_arrays/Patient_id.npy
  np_arrays/DATESTAMP.npy
  np_arrays/TREATED.npy
  lists/X_labels.txt
  lists/Y_labels.txt
```

`TREATED` is 1 for cases and 0 for controls. Arrays share row order. Static features are gender indicators, `Age`, `ADI_NATRANK`, `Mean Income`, `nonwhite_or_hispanic`, and `no bachelor degree`. Clinical/proxy labels use `<outcome>_Day_<start>_<end>`, with `N` denoting a negative day. X covers 48 bins from −1440 through −1; Y contains day zero, days 1–29, and 23 subsequent 30-day bins through day 719. Supply follow-up claim counts, not pre-binarized Y values. The analysis restricts the clinical feature set to the final manifest.

## Supporting notebook inputs

The summary-table notebook also uses static ZIP information and the Census linkage table. The geographic notebook expects `DECENNIALDHC2020.P1-Data.csv` with the source two-row Census headers and geographic export CSVs under `data/komodo_geographic/` containing `PATIENT_ID` and `PATIENT_ZIP3`. The ICD-table notebook consumes the final processed code table above.

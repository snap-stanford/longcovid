# Pipeline guide

Run commands from the repository root and set paths before launching Python or Jupyter:

```bash
export LONGCOVID_HOME=/secure/path/to/longcovid
```

The workspace contains `data/` and `outputs/`. Individual locations can be overridden with `LONGCOVID_DATA`, `LONGCOVID_RAW_DATA`, `LONGCOVID_OUTPUTS`, and `LONGCOVID_CENSUS`.

## 1. Supply final study inputs

Provide the claims exports, per-design cohort membership, final expanded ICD code table, final 614-outcome manifest, and final ADI/Census linkage tables described in [DATA.md](DATA.md). Cohort construction, diagnosis-list curation, and population-weighted reference generation are upstream inputs; earlier implementations of those missing steps are not included.

## 2. Preprocess the claims

For each design, provide its verified enrollment export directory and consistent patient-ID mapping. The matching membership CSV contains `PATIENT_ID` and binary `TREATMENT`:

```bash
python preprocess_pipeline.py -np 8 -sl \
  -co cohort_test_based_control \
  -ptf "$LONGCOVID_HOME/data/raw_data/ptf_files/test_based_control.csv" \
  -fl test_based_control -bf batches_test_based_control
```

Use the corresponding names `unrestricted_control` or `negative_control_exposure` for the other designs. Output directories match the three supplied configurations. The preprocessing consumes the final code definitions directly, creates monthly diagnosis/visit/payer-complete arrays, joins static covariates, selects membership rows, and links the supplied Census values. It produces `<study>_census` under `data/raw_data/batches_<study>/`.

For distributed medical-header processing, first run the same command with `--prepare-only` on one server, then with `-ms` on each worker. Finally rerun the ordinary command on a single server. All workers need identical inputs and a shared filesystem. Use a fresh batch directory after changing inputs; completion markers resume existing intermediates.

## 3. Estimate effects

```bash
python baseline_pipeline.py -c test_based_control.yaml -nc 4
python baseline_pipeline.py -c unrestricted_control.yaml -nc 4
python baseline_pipeline.py -c negative_control_exposure.yaml -nc 4
```

Each command fits the manuscript's constrained balancing objective and runs 1,000 patient bootstrap replicates. `-nc 1` uses less worker memory. Outputs under `outputs/<study>/` include:

- `weights.pkl`: fitted patient-linked control weights.
- `run_metadata.json`: resolved configuration, package versions, manifest checksum, and example status.
- `bootstrap_dataframes/0.csv`: four-window point estimates; `1.csv` through `1000.csv` contain resamples.
- `timeseries/bootstrap_dataframes/`: pre-enrollment and follow-up monthly means for the same resamples.
- `timeseries/means.csv`: case, weighted-control, and unweighted-control means.
- `risk_estimates.csv`: RR, monthly RD, ordinary/adjusted intervals, and two-sided tests.

Column construction, count handling, normalization, and inference are defined in [METHODS.md](METHODS.md). Study configurations select input paths and solver only. Unknown analysis settings are rejected.

## 4. Export figures

```bash
python plot_results.py \
  "$LONGCOVID_HOME/outputs/test_based_control" \
  "$LONGCOVID_HOME/outputs/unrestricted_control" \
  "$LONGCOVID_HOME/outputs/negative_control_exposure" \
  --output "$LONGCOVID_HOME/outputs/figures" \
  --outcomes Myopathy Hair_loss Pneumothorax
```

The command exports attributable-outcome count comparisons, RR/RD panels for all retained outcomes, and optional monthly trajectories using the first study directory. Figure source data are saved alongside the figures. Monthly trajectories use ordinary Gaussian 95% intervals; window panels use Bonferroni-adjusted intervals.

## 5. Supporting notebooks

Install `requirements-notebooks.txt`. Set `LONGCOVID_STUDY` to the selected design before opening `notebooks/SummaryTable.ipynb`; it writes demographic and healthcare-utilization tables as CSV and LaTeX. `CensusKomodoBiasFigure.ipynb` compares ZIP-3 geographic representation and additionally needs the specified Census/geographic inputs. `DiagnosisICDTable.ipynb` formats the final processed ICD mapping as LaTeX. All notebooks are delivered with outputs cleared.

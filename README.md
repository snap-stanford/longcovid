# Long COVID study: analysis code

Code accompanying **Revised estimates of types and duration of long COVID symptoms based on claims records from 245 million US patients**, by Hamed Nilforoshan, Julia Reisler, Erfan Jahanparast, Michael Moor, Steven N. Goodman, Stefan Wager, and Jure Leskovec (PNAS).

The workflow prepares monthly patient histories, fits synthetic-control weights, estimates effects over four follow-up windows, and exports figures and supporting tables. Three configurations select the test-based control, unrestricted control, and negative-control exposure designs. All three use the same final-method analysis implementation.

## System requirements

Use Python 3.10 and Conda on Linux or macOS. The checked-in [environment.yml](environment.yml) installs the analysis packages, tests, and Jupyter; the scientific package versions are pinned in [requirements-lock.txt](requirements-lock.txt). Run the commands below from the repository root in a Bash or Zsh terminal.

The synthetic example runs on a CPU and uses the bundled CLARABEL solver. Study-data analyses use MOSEK and require a valid MOSEK license configured before running the analysis. Full-study memory and runtime requirements depend on the supplied claims exports and have not been benchmarked for this release. Use `-nc 1` to reduce bootstrap worker memory.

## Computing basic results with simulated data

Patient-level study data are supplied separately. The following steps run the available analysis with artificial data and produce example results.

1. Create and activate the environment:

   ```bash
   conda env create --file environment.yml
   conda activate longcovid-paper
   ```

2. Run the unit and integration tests:

   ```bash
   python -m pytest -q
   ```

   The expected result is **13 passed**. These tests cover the final-method transformations and inference, plus the synthetic workflow with one and two workers.

3. Choose a fresh workspace and generate simulated data:

   ```bash
   export LONGCOVID_HOME="$PWD/example_workspace"
   python make_example.py
   ```

   This creates 100 artificial patients, two clinical outcomes, covariates, and an example outcome manifest under `$LONGCOVID_HOME/data/example/`. Use a new workspace for each run; existing example data and analysis outputs are protected from overwriting.

4. Fit control weights and estimate effects:

   ```bash
   python baseline_pipeline.py -c example.yaml --example --n-bootstraps 20 -nc 1
   ```

   This uses the study's covariate construction, balancing objective, and inference with 20 bootstrap replicates. The study configurations use 1,000 replicates.

5. Export figures and their source tables:

   ```bash
   python plot_results.py "$LONGCOVID_HOME/outputs/synthetic_example" \
     --output "$LONGCOVID_HOME/outputs/example_figures" \
     --outcomes Example_respiratory_condition Example_negative_control
   ```

6. Inspect the effect estimates:

   ```bash
   python - <<'PY'
   import os
   from pathlib import Path
   import pandas as pd

   results = Path(os.environ["LONGCOVID_HOME"]) / "outputs/synthetic_example"
   estimates = pd.read_csv(results / "risk_estimates.csv")
   print(f"Effect-estimate rows: {len(estimates)}")
   print(estimates[["OUTCOME", "window", "RR", "RD"]].to_string(index=False))
   PY
   ```

   Expect **8 effect-estimate rows**: two outcomes across days 0–29, 30–119, 120–359, and 360–719. The results directory also contains `weights.pkl`, `run_metadata.json`, bootstrap tables, and monthly means. The figure directory contains `attributable_outcome_counts.pdf`, four window-panel PDFs, `monthly_trajectories.pdf`, PNG previews, and CSV source tables. Simulated values illustrate the computation and have no clinical interpretation.

## Computing results with study data

1. Activate the same environment and select the workspace containing the authorized study inputs:

   ```bash
   conda activate longcovid-paper
   export LONGCOVID_HOME=/secure/path/to/longcovid
   ```

   Supply the verified cohort/claims exports, final expanded ICD mapping, 614-outcome manifest (including 49 negative controls), and final Census/ADI linkage tables using the paths and schemas in [docs/DATA.md](docs/DATA.md). The final cohort-construction/export implementation and several supplementary analyses are unavailable; see [docs/AVAILABILITY.md](docs/AVAILABILITY.md) for the complete scope. Analyses can also start from verified prepared cohorts using the layout in the data guide.

2. Prepare monthly histories for each design:

   ```bash
   for study in test_based_control unrestricted_control negative_control_exposure; do
     python preprocess_pipeline.py -np 8 -sl \
       -co "cohort_${study}" \
       -ptf "$LONGCOVID_HOME/data/raw_data/ptf_files/${study}.csv" \
       -fl "$study" -bf "batches_${study}"
   done
   ```

   Each command writes a prepared cohort to `data/raw_data/batches_<study>/<study>_census/` under the workspace. [docs/PIPELINE.md](docs/PIPELINE.md) explains preprocessing, shared-filesystem execution, and resuming intermediate work.

3. Fit weights and run the 1,000-replicate bootstrap for each design:

   ```bash
   python baseline_pipeline.py -c test_based_control.yaml -nc 4
   python baseline_pipeline.py -c unrestricted_control.yaml -nc 4
   python baseline_pipeline.py -c negative_control_exposure.yaml -nc 4
   ```

   Results are written to `outputs/<study>/` under the workspace. Each `risk_estimates.csv` contains risk ratios, monthly risk differences, confidence intervals, and tests for the four manuscript windows. The objective and inference definitions are in [docs/METHODS.md](docs/METHODS.md).

4. Export comparisons across the three designs and selected monthly trajectories:

   ```bash
   python plot_results.py \
     "$LONGCOVID_HOME/outputs/test_based_control" \
     "$LONGCOVID_HOME/outputs/unrestricted_control" \
     "$LONGCOVID_HOME/outputs/negative_control_exposure" \
     --output "$LONGCOVID_HOME/outputs/figures" \
     --outcomes Myopathy Hair_loss Pneumothorax
   ```

   Window panels include all retained outcomes; `--outcomes` selects the monthly trajectories from the first study directory. Use outcome identifiers from the final manifest.

5. Generate supporting descriptive tables and geographic comparisons:

   ```bash
   export LONGCOVID_STUDY=test_based_control
   jupyter notebook notebooks/
   ```

   Run `SummaryTable.ipynb` for demographic/utilization tables, `CensusKomodoBiasFigure.ipynb` for geographic representation, and `DiagnosisICDTable.ipynb` for the ICD-code table. Their additional input requirements are documented in the data and pipeline guides.

Validation details are in [docs/VALIDATION.md](docs/VALIDATION.md); article attribution is in [CITATION.cff](CITATION.cff).

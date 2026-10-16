# Long COVID study: analysis code

Code for the analyses in **Revised estimates of types and duration of long COVID symptoms based on claims records from 245 million US patients** (PNAS). Author information is provided in [CITATION.cff](CITATION.cff).

## System requirements

The code uses Python 3.10 in a Conda environment on Linux or macOS. [environment.yml](environment.yml) includes the analysis packages, tests, and Jupyter. Scientific package versions are pinned in [requirements-lock.txt](requirements-lock.txt).

The simulated example runs on a CPU using the CLARABEL solver. Analyses of the study data use MOSEK, which requires a license. Memory and runtime for the full analysis depend on the size of the claims exports; full-study benchmarks are not available. Bootstrap computation can use multiple workers, with `-nc 1` available to reduce worker memory.

## Computing basic results

Run the following commands from the repository root in a Bash or Zsh terminal.

1. Set up the environment. Create the Conda environment using the supplied file, then activate it before running any code.

   ```bash
   conda env create --file environment.yml
   conda activate longcovid-paper
   ```

2. Run the tests. These check the main data transformations, weighting, and statistical calculations, as well as the simulated analysis with one and two workers. The expected result is **13 passed**.

   ```bash
   python -m pytest -q
   ```

3. Generate a simulated cohort. The analysis uses non-public claims data. We provide a script that generates artificial data so the code can be run without access to the study records. Set `LONGCOVID_HOME` to a fresh directory for the example.

   ```bash
   export LONGCOVID_HOME="$PWD/example_workspace"
   python make_example.py
   ```

   This creates 100 artificial patients, two clinical outcomes, and the covariates needed for the analysis under `$LONGCOVID_HOME/data/example/`. Choose a new workspace when repeating the example, since existing data and analysis outputs are protected from overwriting.

4. Compute the effect estimates. This fits control weights and estimates effects in the four follow-up windows described in the paper. The example uses the same covariate construction, balancing objective, and inference code, with 20 bootstrap replicates rather than the 1,000 used for the study data.

   ```bash
   python baseline_pipeline.py -c example.yaml --example --n-bootstraps 20 -nc 1
   ```

5. Make figures. This writes outcome-count comparisons, risk-ratio and risk-difference panels, and monthly trajectories for the selected outcomes. The figure source data are saved as CSV files in the same directory.

   ```bash
   python plot_results.py "$LONGCOVID_HOME/outputs/synthetic_example" \
     --output "$LONGCOVID_HOME/outputs/example_figures" \
     --outcomes Example_respiratory_condition Example_negative_control
   ```

6. Explore the results. The following code loads the effect estimates and displays the risk ratio (`RR`) and monthly risk difference (`RD`) for each outcome and follow-up window.

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

   With simulated data, the table has **8 rows**: two outcomes in each of the windows covering days 0–29, 30–119, 120–359, and 360–719. These values illustrate the computation and have no clinical interpretation. The results directory also contains the fitted weights, run metadata, bootstrap tables, and monthly means. The figures include `attributable_outcome_counts.pdf`, four window-panel PDFs, and `monthly_trajectories.pdf`, with PNG previews and CSV source tables.

## Computing results with study data

These analyses require the study cohort and claims exports, the final expanded ICD mapping, the 614-outcome manifest (including 49 negative controls), and the Census/ADI linkage tables. Input paths and formats are described in [docs/DATA.md](docs/DATA.md). The available analyses and missing materials are listed in [docs/AVAILABILITY.md](docs/AVAILABILITY.md).

1. Set the data directory. Activate the same environment and set `LONGCOVID_HOME` to the directory containing the study inputs.

   ```bash
   conda activate longcovid-paper
   export LONGCOVID_HOME=/secure/path/to/longcovid
   ```

2. Prepare patient histories. This converts the claims exports into monthly diagnosis and utilization histories and adds the static and Census covariates. Run the preprocessing for each of the three study designs.

   ```bash
   for study in test_based_control unrestricted_control negative_control_exposure; do
     python preprocess_pipeline.py -np 8 -sl \
       -co "cohort_${study}" \
       -ptf "$LONGCOVID_HOME/data/raw_data/ptf_files/${study}.csv" \
       -fl "$study" -bf "batches_${study}"
   done
   ```

   Prepared cohorts are written to `data/raw_data/batches_<study>/<study>_census/` under the workspace. See [docs/PIPELINE.md](docs/PIPELINE.md) for details on shared-filesystem execution and resuming intermediate work. If verified prepared cohorts are already available in the format described in the data guide, start with the next step.

3. Compute effects for each study design. The three configurations use the same weighting and bootstrap calculations for the test-based control, unrestricted control, and negative-control exposure analyses. Each command runs 1,000 bootstrap replicates.

   ```bash
   python baseline_pipeline.py -c test_based_control.yaml -nc 4
   python baseline_pipeline.py -c unrestricted_control.yaml -nc 4
   python baseline_pipeline.py -c negative_control_exposure.yaml -nc 4
   ```

   Each analysis writes its results to `outputs/<study>/` under the workspace. The `risk_estimates.csv` file contains risk ratios, monthly risk differences, confidence intervals, and statistical tests for the four follow-up windows. Details of the weighting objective and inference are in [docs/METHODS.md](docs/METHODS.md).

4. Make comparison figures. This compares the three study designs and plots monthly trajectories for the selected outcomes from the first study directory. The window panels include all retained outcomes; change `--outcomes` to choose the monthly trajectories, using identifiers from the final manifest.

   ```bash
   python plot_results.py \
     "$LONGCOVID_HOME/outputs/test_based_control" \
     "$LONGCOVID_HOME/outputs/unrestricted_control" \
     "$LONGCOVID_HOME/outputs/negative_control_exposure" \
     --output "$LONGCOVID_HOME/outputs/figures" \
     --outcomes Myopathy Hair_loss Pneumothorax
   ```

5. Compute supporting tables and geographic comparisons. Set the study design and open the supplied notebooks.

   ```bash
   export LONGCOVID_STUDY=test_based_control
   jupyter notebook notebooks/
   ```

   `SummaryTable.ipynb` produces demographic and healthcare-utilization tables. `CensusKomodoBiasFigure.ipynb` compares geographic representation, and `DiagnosisICDTable.ipynb` produces the ICD-code table. Additional inputs for these notebooks are described in the data and pipeline guides.

See [docs/VALIDATION.md](docs/VALIDATION.md) for the environment and validation results.

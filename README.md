# Long COVID study: analysis code

Code accompanying **Revised estimates of types and duration of long COVID symptoms based on claims records from 245 million US patients**, by Hamed Nilforoshan, Julia Reisler, Erfan Jahanparast, Michael Moor, Steven N. Goodman, Stefan Wager, and Jure Leskovec.

This release contains one analysis path following the final manuscript: monthly pre-enrollment covariates, constrained synthetic-control weighting, four follow-up windows, and Gaussian bootstrap inference with Bonferroni correction. Three configurations select the test-based control design, unrestricted control design, and negative-control exposure design. They use the same analysis implementation.

## Install

Use Python 3.10 on Linux or macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The study configurations use the MOSEK solver and require a license. The synthetic example uses CLARABEL with the same mathematical objective. `requirements-lock.txt` contains the complete dependency resolution used for validation. For notebooks, install `requirements-notebooks.txt`.

## Study workflow

| Step | Entry point | Result |
| --- | --- | --- |
| 1. Prepare monthly histories from verified claims/cohort exports | `preprocess_pipeline.py` | Sparse pre-enrollment histories and follow-up counts |
| 2. Select final covariates, fit weights, and estimate effects | `baseline_pipeline.py` | Weights, bootstrap means, risk ratios, and monthly risk differences |
| 3. Generate figures and source tables | `plot_results.py` | Outcome-count comparisons, RR/RD panels, and monthly trajectories |
| 4. Generate descriptive supporting artifacts | `notebooks/` | Summary tables, geographic comparison, and ICD-code table |

Follow [the pipeline guide](docs/PIPELINE.md). [Input specifications](docs/DATA.md), [method definitions](docs/METHODS.md), and [availability notes](docs/AVAILABILITY.md) describe the supported workflow and missing materials. Patient-level data are not included.

## Synthetic example

```bash
export LONGCOVID_HOME=/tmp/longcovid-example
python make_example.py
python baseline_pipeline.py -c example.yaml --example --n-bootstraps 20 -nc 1
python plot_results.py "$LONGCOVID_HOME/outputs/synthetic_example" \
  --output "$LONGCOVID_HOME/outputs/example_figures" \
  --outcomes Example_respiratory_condition Example_negative_control
```

The example uses artificial claim counts and the same covariate, objective, and inference code as the study workflow. Existing nonempty run directories are not overwritten.

## Tests

```bash
pip install -r requirements-test.txt
python -m pytest -q
```

See [validation results](docs/VALIDATION.md). Article attribution is provided in `CITATION.cff`.

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from baseline_pipeline import run
from make_example import build_example_cohort
from preprocess_pipeline import (
    day_to_month_category,
    get_outcome_df,
    read_combine_batch_tsvs,
)
from src.baseline_pipeline.convex_script import fit_control_weights
from src.baseline_pipeline.transformation_script import (
    WINDOWS,
    aggregate_outcome_windows,
    prepare_analysis_data,
)
from src.analysis import (
    align_control_weights,
    normal_inference,
    read_outcome_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_only_the_four_manuscript_windows_are_used():
    labels = ["Outcome_Day_0_1", "Outcome_Day_1_30"] + [
        f"Outcome_Day_{day}_{day + 30}" for day in range(30, 720, 30)
    ]
    matrix, names = aggregate_outcome_windows(
        csr_matrix([np.arange(25)]), labels, ["Outcome"]
    )
    assert names == [
        "Outcome_Window_0_30",
        "Outcome_Window_30_120",
        "Outcome_Window_120_360",
        "Outcome_Window_360_720",
    ]
    np.testing.assert_array_equal(
        matrix.toarray(), [[1, 2 + 3 + 4, sum(range(5, 13)), sum(range(13, 25))]]
    )
    assert len(WINDOWS) == 4
    assert day_to_month_category(29) == "Day_1_30"
    assert day_to_month_category(30) == "Day_30_60"
    assert day_to_month_category(719) == "Day_690_720"


def test_claim_counts_are_preserved_while_pre_enrollment_diagnoses_are_binary():
    data, outcomes = build_example_cohort()
    prepared = prepare_analysis_data(data, outcomes)
    columns = [
        i
        for i, label in enumerate(prepared["X_labels"])
        if label.startswith(outcomes[0] + "_Day_")
    ]
    assert prepared["X"][:, columns].max() == 1
    assert prepared["Y"].max() > 1
    assert (
        prepared["X"][
            :, prepared["X_labels"].index("Outpatient_Visit_Day_N1440_N1410")
        ].max()
        > 1
    )
    assert not any("Visit" in label for label in prepared["Y_labels"])


def test_mean_imputation_for_age_and_adi():
    data, outcomes = build_example_cohort()
    matrix = data["X"].toarray()
    matrix[0, data["X_labels"].index("Age")] = np.nan
    matrix[1, data["X_labels"].index("ADI_NATRANK")] = np.nan
    data["X"] = csr_matrix(matrix)
    prepared = prepare_analysis_data(data, outcomes)
    assert prepared["X"][0, 3] == pytest.approx(np.nanmean(matrix[:, 3]))
    assert prepared["X"][1, 4] == pytest.approx(np.nanmean(matrix[:, 4]))


def test_balancing_fits_the_unregularized_simplex_objective():
    weights = fit_control_weights(
        csr_matrix([[0.0], [2.0]]), np.array([0.5]), solver="CLARABEL"
    )
    np.testing.assert_allclose(weights, [0.75, 0.25], atol=1e-5)


def test_unsupported_configuration_settings_are_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("name: example\nunsupported_setting: true\n")
    with pytest.raises(ValueError, match="unsupported analysis settings"):
        run(path)


def test_gaussian_intervals_and_bonferroni_use_point_estimate():
    result = normal_inference(np.array([2.0]), np.array([[1.0], [3.0]]), 1.0, 2)
    assert result["SE"][0] == 1
    assert result["CI_LOWER_95"][0] == pytest.approx(0.04003601546)
    assert result["CI_UPPER_95"][0] == pytest.approx(3.95996398454)
    assert result["p_value"][0] == pytest.approx(0.31731050786)
    assert result["p_bonferroni"][0] == pytest.approx(0.63462101572)


def test_undefined_bootstrap_ratio_is_not_dropped():
    result = normal_inference(np.array([2.0]), np.array([[1.0], [np.inf]]), 1.0, 614)
    assert not result["valid"][0]
    assert np.isnan(result["p_value"][0])


def test_control_weights_align_by_token_and_reject_missing_patients():
    data = {
        "TREATED": np.array([1, 0, 0]),
        "Patient_token": np.array(["case", "b", "a"]),
    }
    rows, weights = align_control_weights(data, [0.2, 0.8], ["a", "b"])
    np.testing.assert_array_equal(rows, [1, 2])
    np.testing.assert_allclose(weights, [0.8, 0.2])
    with pytest.raises(ValueError, match="outside"):
        align_control_weights(data, [1.0], ["missing"])


def test_exact_icd_matching_does_not_match_code_prefixes():
    records = pd.DataFrame(
        {"DIAGNOSIS_ARRAY": ['["R100"]', '["R1000"]', '["R100", "I10"]']}
    )
    assert get_outcome_df(records, ["R100"]).index.tolist() == [0, 2]


def test_threaded_export_reader_reads_each_file_and_propagates_failure(tmp_path):
    paths = []
    for value in [1, 2, 3]:
        path = tmp_path / f"{value}.tsv"
        path.write_text(f"PATIENT_ID\n{value}\n")
        paths.append(path)
    assert read_combine_batch_tsvs(paths, ["PATIENT_ID"], None, None)[
        "PATIENT_ID"
    ].tolist() == [1, 2, 3]
    with pytest.raises(FileNotFoundError):
        read_combine_batch_tsvs(
            paths + [tmp_path / "missing.tsv"], ["PATIENT_ID"], None, None
        )


def test_reduced_outcome_manifest_requires_explicit_example_mode(tmp_path):
    path = tmp_path / "outcomes.csv"
    path.write_text("OUTCOME,negative\nExample,true\n")
    with pytest.raises(ValueError, match="614"):
        read_outcome_manifest(path)
    assert read_outcome_manifest(path, example=True)["negative"].tolist() == [True]


@pytest.mark.parametrize("workers", [1, 2])
def test_synthetic_pipeline_and_figures(tmp_path, workers):
    env = os.environ.copy()
    env["LONGCOVID_HOME"] = str(tmp_path / "workspace")
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    for key in ["LONGCOVID_DATA", "LONGCOVID_RAW_DATA", "LONGCOVID_OUTPUTS"]:
        env.pop(key, None)

    def command(*args):
        result = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    command("make_example.py")
    command(
        "baseline_pipeline.py",
        "-c",
        "example.yaml",
        "--example",
        "--n-bootstraps",
        "5",
        "-nc",
        str(workers),
    )
    output = Path(env["LONGCOVID_HOME"]) / "outputs" / "synthetic_example"
    estimates = pd.read_csv(output / "risk_estimates.csv")
    assert len(estimates) == 8
    assert set(estimates["window"]) == {"0_30", "30_120", "120_360", "360_720"}
    assert (estimates["replicates"] == 5).all()
    assert json.loads((output / "run_metadata.json").read_text())["synthetic"] is True
    command(
        "plot_results.py",
        str(output),
        "--output",
        str(tmp_path / "figures"),
        "--outcomes",
        "Example_respiratory_condition",
    )
    assert (tmp_path / "figures" / "attributable_outcome_counts.pdf").is_file()
    assert (tmp_path / "figures" / "monthly_trajectories.csv").is_file()

"""Fixed-weight patient bootstrap and the inference described in Methods M3.

Monthly pre-enrollment diagnosis indicators are used for balancing; follow-up
claim counts are summed within the four analysis windows. Inference uses Gaussian
bootstrap intervals and Bonferroni correction within each window.
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import norm

from .utils.data_utils import load_data_dic
from .baseline_pipeline.transformation_script import WINDOWS, prepare_analysis_data

_bootstrap_cache = None


def initialize_bootstrap(treated, control, weights, labels, output):
    global _bootstrap_cache
    _bootstrap_cache = (treated, control, weights, labels, Path(output))


def bootstrap_iteration(seed):
    """Write one original-estimator replicate; seed zero is the point estimate."""
    treated, control, weights, labels, output = _bootstrap_cache
    path = output / f"{seed}.csv"
    if seed == 0:
        sampled_treated, sampled_control, sampled_weights = treated, control, weights
    else:
        # Preserve the original RandomState reset in each arm.
        treated_rows = np.random.RandomState(seed).choice(
            treated.shape[0], treated.shape[0], replace=True
        )
        control_rows = np.random.RandomState(seed).choice(
            control.shape[0], control.shape[0], replace=True
        )
        sampled_treated = treated[treated_rows]
        sampled_control = control[control_rows]
        sampled_weights = weights[control_rows]
    if sampled_weights.sum() <= 0:
        raise ValueError(f"Bootstrap replicate {seed} has zero total control weight.")
    frame = pd.DataFrame(
        {
            "Y_label": labels,
            "Y_trt": np.asarray(sampled_treated.mean(axis=0)).ravel(),
            "Y_ctrl": np.asarray(sampled_control.T @ sampled_weights).ravel()
            / sampled_weights.sum(),
        }
    )
    frame.to_csv(path, index=False)


def read_outcome_manifest(path, example=False):
    frame = pd.read_csv(path, dtype={"OUTCOME": str})
    required = {"OUTCOME", "negative"}
    if not required.issubset(frame):
        raise ValueError(f"Outcome manifest requires columns {sorted(required)}.")
    if frame["OUTCOME"].isna().any() or frame["OUTCOME"].duplicated().any():
        raise ValueError(
            "Outcome manifest must have unique, nonempty outcome identifiers."
        )
    truth = frame["negative"].astype(str).str.lower()
    if not truth.isin(["true", "false", "1", "0"]).all():
        raise ValueError("negative must contain true/false or 1/0.")
    frame["negative"] = truth.isin(["true", "1"])
    if not example and (len(frame) != 614 or frame["negative"].sum() != 49):
        raise ValueError(
            "The outcome manifest must contain 614 outcomes, including 49 negative controls."
        )
    return frame


def align_control_weights(data, weights, tokens):
    weights, tokens = np.asarray(weights).ravel(), np.asarray(tokens)
    if len(weights) != len(tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Weights must have one unique patient token per entry.")
    if not np.isfinite(weights).all() or weights.min() < -1e-7 or weights.sum() <= 0:
        raise ValueError(
            "Control weights must be finite, nonnegative, and have positive total weight."
        )
    weights = np.maximum(weights, 0)
    control_rows = np.flatnonzero(data["TREATED"] == 0)
    control_tokens = data["Patient_token"][control_rows]
    if len(set(control_tokens)) != len(control_tokens):
        raise ValueError(
            "The analysis cohort contains duplicate control patient tokens."
        )
    if set(tokens) - set(control_tokens):
        raise ValueError(
            "Some saved weights refer to patients outside the control cohort."
        )
    lookup = dict(zip(tokens, weights))
    return control_rows, np.array([lookup.get(token, 0.0) for token in control_tokens])


def evaluate_bootstraps(
    data_path,
    featurized_path,
    weights,
    tokens,
    output,
    outcomes,
    n_bootstraps=1000,
    num_cores=1,
):
    """Export monthly and window-level means using identical patient resamples."""
    output = Path(output)
    raw = prepare_analysis_data(load_data_dic(data_path), outcomes)
    bucketed = load_data_dic(featurized_path)
    if not np.array_equal(raw["Patient_token"], bucketed["Patient_token"]):
        raise ValueError("Raw and featurized cohorts have different patient order.")
    control_rows, aligned = align_control_weights(raw, weights, tokens)
    treated_rows = np.flatnonzero(raw["TREATED"] == 1)
    if not len(treated_rows) or not len(control_rows):
        raise ValueError("Both study arms must contain patients.")
    monthly = raw
    monthly_matrix = hstack([monthly["X"], monthly["Y"]], format="csr")
    monthly_labels = monthly["X_labels"] + monthly["Y_labels"]
    all_matrix = hstack([monthly_matrix, bucketed["Y"]], format="csr")
    bucket_labels = [f"BUCKET::{label}" for label in bucketed["Y_labels"]]
    combined = output / "bootstrap_combined"
    combined.mkdir(parents=True, exist_ok=True)
    cache = (
        all_matrix[treated_rows],
        all_matrix[control_rows],
        aligned,
        monthly_labels + bucket_labels,
        combined,
    )
    if num_cores == 1:
        initialize_bootstrap(*cache)
        for seed in range(n_bootstraps + 1):
            bootstrap_iteration(seed)
    else:
        # Initializers also work on platforms whose default process start method is spawn.
        with mp.get_context("spawn").Pool(
            num_cores, initializer=initialize_bootstrap, initargs=cache
        ) as pool:
            pool.map(bootstrap_iteration, range(n_bootstraps + 1))
    monthly_dir = output / "timeseries" / "bootstrap_dataframes"
    bucket_dir = output / "bootstrap_dataframes"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(combined.glob("*.csv")):
        frame = pd.read_csv(path)
        is_bucket = frame["Y_label"].str.startswith("BUCKET::")
        frame.loc[~is_bucket].to_csv(monthly_dir / path.name, index=False)
        bucket = frame.loc[is_bucket].copy()
        bucket["Y_label"] = bucket["Y_label"].str.removeprefix("BUCKET::")
        bucket.to_csv(bucket_dir / path.name, index=False)
        path.unlink()
    combined.rmdir()
    means = pd.read_csv(monthly_dir / "0.csv")
    means["Y_ctrl_unweighted"] = np.asarray(
        monthly_matrix[control_rows].mean(axis=0)
    ).ravel()
    means.to_csv(output / "timeseries" / "means.csv", index=False)
    (output / "bootstrap_metadata.json").write_text(
        json.dumps(
            {
                "replicates": n_bootstraps,
                "point_estimate_file": "0.csv",
                "seed_range": [1, n_bootstraps],
                "weights_refit_per_replicate": False,
                "treated_patients": len(treated_rows),
                "control_patients": len(control_rows),
            },
            indent=2,
        )
        + "\n"
    )


def normal_inference(point, samples, null, comparisons):
    """Normal intervals centered on the observed estimate; ddof=0 matches the source."""
    point, samples = np.asarray(point), np.asarray(samples)
    valid = np.isfinite(samples).all(axis=0) & np.isfinite(point)
    standard_error = np.full(point.shape, np.nan)
    standard_error[valid] = np.std(samples[:, valid], axis=0, ddof=0)
    p_value = np.full(point.shape, np.nan)
    positive = valid & (standard_error > 0)
    p_value[positive] = 2 * norm.sf(
        np.abs((point[positive] - null) / standard_error[positive])
    )
    zero = valid & (standard_error == 0)
    p_value[zero] = np.where(point[zero] == null, 1.0, 0.0)
    critical = norm.ppf(1 - 0.05 / (2 * comparisons))
    return {
        "SE": standard_error,
        "CI_LOWER_95": point - norm.ppf(0.975) * standard_error,
        "CI_UPPER_95": point + norm.ppf(0.975) * standard_error,
        "CI_LOWER_BONFERRONI": point - critical * standard_error,
        "CI_UPPER_BONFERRONI": point + critical * standard_error,
        "p_value": p_value,
        "p_bonferroni": np.minimum(1.0, p_value * comparisons),
        "valid": valid,
    }


def summarize_windows(bootstrap_path, manifest):
    """Return RR and monthly RD for the four prespecified analysis windows."""
    folder = Path(bootstrap_path)
    paths = sorted(
        (p for p in folder.glob("*.csv") if p.stem.isdigit()), key=lambda p: int(p.stem)
    )
    if not paths or paths[0].stem != "0" or len(paths) < 3:
        raise ValueError(
            "Point estimates and at least two bootstrap replicates are required."
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path).set_index("Y_label")
        if frame.index.duplicated().any():
            raise ValueError(f"Duplicate outcome labels in {path}.")
        frames.append(frame)
    results = []
    for window, (_, _, months) in WINDOWS.items():
        labels = [f"{outcome}_Window_{window}" for outcome in manifest["OUTCOME"]]
        arrays = [frame.loc[labels, ["Y_trt", "Y_ctrl"]].to_numpy() for frame in frames]
        values = np.stack(arrays)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = values[:, :, 0] / values[:, :, 1]
        rd = (values[:, :, 0] - values[:, :, 1]) / months
        frame = manifest.copy()
        frame["window"] = window
        frame["Y_trt"], frame["Y_ctrl"] = values[0, :, 0], values[0, :, 1]
        frame["RR"], frame["RD"] = rr[0], rd[0]
        for prefix, statistic, null in [("RR", rr, 1.0), ("RD", rd, 0.0)]:
            for field, result in normal_inference(
                statistic[0], statistic[1:], null, len(manifest)
            ).items():
                frame[f"{prefix}_{field}"] = result
        frame["attributable"] = (
            (frame["RR"] >= 1.1) & (frame["RR_p_bonferroni"] < 0.05) & frame["RR_valid"]
        )
        frame["confident_RR_at_least_1_1"] = (
            frame["RR_CI_LOWER_BONFERRONI"] >= 1.1
        ) & frame["RR_valid"]
        frame["replicates"] = len(paths) - 1
        results.append(frame)
    result = pd.concat(results, ignore_index=True)
    if (~result["RR_valid"]).any():
        logging.warning(
            "Some risk ratios are undefined in at least one replicate; their intervals and tests are left undefined."
        )
    return result

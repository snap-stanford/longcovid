"""Run one named study design from prepared sparse cohorts to bootstrap summaries."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import yaml

from src.baseline_pipeline.convex_script import fit_control_weights
from src.baseline_pipeline.transformation_script import get_matrices
from src.publication_analysis import (
    evaluate_bootstraps,
    read_outcome_manifest,
    summarize_windows,
)
from src.utils.constants import BASELINE_DATA_PATH, DATA_PATH, OUTPUT_DIR, REPO_ROOT
from src.utils.data_utils import writePickle


def resolve_config_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    return REPO_ROOT / "configs" / path


def run(config_path, num_cores=1, n_bootstraps=1000, example=False):
    config_path = resolve_config_path(config_path)
    config = yaml.safe_load(config_path.read_text())
    allowed_keys = {"name", "data_path", "outcome_manifest", "solver", "synthetic"}
    if set(config) - allowed_keys:
        raise ValueError(
            "The configuration contains settings outside the final manuscript workflow."
        )
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = Path(DATA_PATH) / data_path
    manifest_path = Path(config["outcome_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = Path(DATA_PATH) / manifest_path
    manifest = read_outcome_manifest(manifest_path, example=example)
    if not example and n_bootstraps != 1000:
        raise ValueError(
            "Use 1,000 replicates for publication analyses. The --example flag is for synthetic validation only."
        )
    if n_bootstraps < 2 or num_cores < 1:
        raise ValueError(
            "At least two bootstrap replicates and one worker are required."
        )
    name = config["name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Study name must be a single directory name.")
    if example and config.get("synthetic") is not True:
        raise ValueError(
            "--example requires a configuration explicitly marked synthetic: true."
        )
    if config.get("synthetic") and not example:
        raise ValueError("The synthetic configuration requires --example.")
    output = Path(OUTPUT_DIR) / name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new name or move the previous run."
        )
    output.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        (str(data_path.resolve()) + json.dumps(config, sort_keys=True)).encode()
    ).hexdigest()[:16]
    feature_name = f"{name}_{signature}"
    feature_path = Path(BASELINE_DATA_PATH) / feature_name
    outcomes = manifest["OUTCOME"].tolist()
    control, treated_mean, tokens = get_matrices(str(data_path), feature_name, outcomes)
    weights = fit_control_weights(control, treated_mean, solver=config["solver"])
    del control, treated_mean
    writePickle(
        {"weights": weights, "patient_tokens": tokens}, str(output / "weights.pkl")
    )
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "cvxpy", "Mosek"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "configuration": config,
                "data_path": str(data_path),
                "outcome_manifest": str(manifest_path),
                "outcome_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "synthetic": example,
                "versions": versions,
                "n_bootstraps": n_bootstraps,
                "inference": "Gaussian bootstrap standard errors; two-sided tests; Bonferroni within window",
            },
            indent=2,
        )
        + "\n"
    )
    evaluate_bootstraps(
        str(data_path),
        str(feature_path),
        weights,
        tokens,
        output,
        outcomes,
        n_bootstraps,
        num_cores,
    )
    summary = summarize_windows(output / "bootstrap_dataframes", manifest)
    summary.to_csv(output / "risk_estimates.csv", index=False)
    logging.info("Completed analysis: %s", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Configuration path or filename under configs/.",
    )
    parser.add_argument("-nc", "--num-cores", type=int, default=1)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument(
        "--example",
        action="store_true",
        help="Validate synthetic data with fewer outcomes and replicates.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(args.config, args.num_cores, args.n_bootstraps, args.example)


if __name__ == "__main__":
    main()

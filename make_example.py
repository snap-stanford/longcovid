"""Generate artificial inputs that exercise the single manuscript analysis path."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.baseline_pipeline.transformation_script import (
    STATIC_COVARIATES,
    VISIT_COVARIATES,
)
from src.utils.constants import DATA_PATH
from src.utils.data_utils import save_data_dic


def build_example_cohort():
    rng = np.random.RandomState(42)
    n = 100
    outcomes = ["Example_respiratory_condition", "Example_negative_control"]
    before = [f"Day_N{abs(day)}_N{abs(day + 30)}" for day in range(-1440, 0, 30)]
    after = ["Day_0_1", "Day_1_30"] + [
        f"Day_{day}_{day + 30}" for day in range(30, 720, 30)
    ]
    groups = outcomes + VISIT_COVARIATES
    static = np.column_stack(
        [
            np.arange(n) % 2,
            1 - np.arange(n) % 2,
            np.zeros(n),
            rng.randint(18, 80, n),
            rng.randint(1, 101, n),
            rng.randint(30000, 150000, n),
            rng.uniform(0.05, 0.95, n),
            rng.uniform(0.1, 0.9, n),
        ]
    )
    x_labels = STATIC_COVARIATES + [
        f"{outcome}_{month}" for outcome in groups + ["closed"] for month in before
    ]
    x = np.column_stack(
        [static, rng.poisson(1.5, (n, len(groups) * 48)), rng.binomial(1, 0.9, (n, 48))]
    )
    y_labels = [f"{outcome}_{month}" for outcome in groups for month in after]
    data = {
        "X": csr_matrix(x),
        "X_labels": x_labels,
        "Y": csr_matrix(rng.poisson(1.5, (n, len(y_labels)))),
        "Y_labels": y_labels,
        "Patient_token": np.array([f"synthetic_{i:03d}" for i in range(n)]),
        "Patient_id": np.arange(n),
        "DATESTAMP": np.full(n, np.datetime64("2020-06-01")),
        "TREATED": np.r_[np.ones(20), np.zeros(n - 20)],
    }
    return data, outcomes


def main():
    root = Path(DATA_PATH) / "example"
    if root.exists():
        raise FileExistsError(f"Example data already exists: {root}")
    data, outcomes = build_example_cohort()
    save_data_dic(data, str(root / "cohort"))
    pd.DataFrame({"OUTCOME": outcomes, "negative": [False, True]}).to_csv(
        root / "analysis_outcomes.csv", index=False
    )
    print(f"Wrote synthetic example to {root}")


if __name__ == "__main__":
    main()

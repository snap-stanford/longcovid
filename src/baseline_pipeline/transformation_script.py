"""Monthly covariates and the four outcome windows specified by the manuscript."""

import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize

from ..utils.constants import BASELINE_DATA_PATH
from ..utils.data_utils import load_data_dic, save_data_dic

WINDOWS = {
    "0_30": (0, 30, 1),
    "30_120": (30, 120, 3),
    "120_360": (120, 360, 8),
    "360_720": (360, 720, 12),
}
STATIC_COVARIATES = [
    "GENDER_F",
    "GENDER_M",
    "GENDER_U",
    "Age",
    "ADI_NATRANK",
    "Mean Income",
    "nonwhite_or_hispanic",
    "no bachelor degree",
]
VISIT_COVARIATES = [
    "Inpatient_Visit",
    "Outpatient_Visit",
    "Home_Visit",
    "Emergency_Room_Visit",
    "Ambulance_Visit",
    "Pharmacy Visit",
    "Non-hospital_institution_Visit",
]
PRE_INTERVALS = [(start, start + 30) for start in range(-1440, 0, 30)]
POST_INTERVALS = [(0, 1), (1, 30)] + [
    (start, start + 30) for start in range(30, 720, 30)
]
LABEL_PATTERN = re.compile(r"^(.*)_Day_(N?\d+)_(N?\d+)$")


def split_label(label):
    match = LABEL_PATTERN.fullmatch(label)
    if not match:
        return None

    def number(value):
        return -int(value[1:]) if value.startswith("N") else int(value)

    return match[1], number(match[2]), number(match[3])


def temporal_columns(labels):
    columns = {}
    for index, label in enumerate(labels):
        key = split_label(label)
        if key is not None:
            if key in columns:
                raise ValueError(f"Duplicate monthly feature: {label}")
            columns[key] = index
    return columns


def select_months(matrix, columns, outcome, intervals):
    keys = [(outcome, start, end) for start, end in intervals]
    missing = [key for key in keys if key not in columns]
    if missing:
        raise ValueError(f"Missing monthly features for {outcome}: {missing[:3]}")
    return matrix[:, [columns[key] for key in keys]].tocsr()


def prepare_analysis_data(data, outcomes):
    """Select the final outcomes, binarize pre-enrollment diagnoses, and keep follow-up counts."""
    x = data["X"].tocsr()
    y = data["Y"].tocsr()
    x_columns = temporal_columns(data["X_labels"])
    y_columns = temporal_columns(data["Y_labels"])
    missing = set(STATIC_COVARIATES) - set(data["X_labels"])
    if missing:
        raise ValueError(f"Missing manuscript covariates: {sorted(missing)}")
    static = (
        x[:, [data["X_labels"].index(label) for label in STATIC_COVARIATES]]
        .toarray()
        .astype(float)
    )
    for label in ["Age", "ADI_NATRANK"]:
        column = STATIC_COVARIATES.index(label)
        missing_values = np.isnan(static[:, column])
        if missing_values.all():
            raise ValueError(
                f"No observed values are available for mean imputation of {label}."
            )
        static[missing_values, column] = np.nanmean(static[:, column])
    if not np.isfinite(static).all():
        raise ValueError(
            "Static covariates other than age/ADI must have finite linked values."
        )
    x_parts = [csr_matrix(static)]
    x_labels = list(STATIC_COVARIATES)
    for outcome in outcomes:
        x_parts.append(select_months(x, x_columns, outcome, PRE_INTERVALS).minimum(1))
        x_labels.extend(
            [f"{outcome}_Day_N{abs(start)}_N{abs(end)}" for start, end in PRE_INTERVALS]
        )
    for outcome in VISIT_COVARIATES + ["closed"]:
        values = select_months(x, x_columns, outcome, PRE_INTERVALS)
        x_parts.append(values.minimum(1) if outcome == "closed" else values)
        x_labels.extend(
            [f"{outcome}_Day_N{abs(start)}_N{abs(end)}" for start, end in PRE_INTERVALS]
        )
    y_parts = [
        select_months(y, y_columns, outcome, POST_INTERVALS) for outcome in outcomes
    ]
    y_labels = [
        f"{outcome}_Day_{start}_{end}"
        for outcome in outcomes
        for start, end in POST_INTERVALS
    ]
    prepared = {
        key: data[key]
        for key in ["Patient_token", "Patient_id", "TREATED", "DATESTAMP"]
        if key in data
    }
    prepared.update(
        X=hstack(x_parts, format="csr"),
        X_labels=x_labels,
        Y=hstack(y_parts, format="csr"),
        Y_labels=y_labels,
    )
    if (
        not np.isfinite(prepared["X"].data).all()
        or not np.isfinite(prepared["Y"].data).all()
    ):
        raise ValueError("Monthly matrices must contain finite values.")
    if (prepared["X"].data < 0).any() or (prepared["Y"].data < 0).any():
        raise ValueError("Monthly counts and covariates must be nonnegative.")
    return prepared


def aggregate_outcome_windows(matrix, labels, outcomes):
    """Sum follow-up claim counts into the four half-open analysis windows."""
    columns = temporal_columns(labels)
    parts, names = [], []
    for outcome in outcomes:
        for window, (start, end, _) in WINDOWS.items():
            intervals = [(a, b) for a, b in POST_INTERVALS if a >= start and b <= end]
            parts.append(
                csr_matrix(
                    select_months(matrix, columns, outcome, intervals).sum(axis=1)
                )
            )
            names.append(f"{outcome}_Window_{window}")
    return hstack(parts, format="csr"), names


def get_matrices(data_path, feature_name, outcomes):
    """Create the single manuscript feature representation and its balancing matrices."""
    prepared = prepare_analysis_data(load_data_dic(data_path), outcomes)
    bucketed = dict(prepared)
    bucketed["Y"], bucketed["Y_labels"] = aggregate_outcome_windows(
        prepared["Y"], prepared["Y_labels"], outcomes
    )
    feature_path = Path(BASELINE_DATA_PATH) / feature_name
    if feature_path.exists():
        raise FileExistsError(f"Feature output already exists: {feature_path}")
    save_data_dic(bucketed, str(feature_path))
    matrix = normalize(prepared["X"], axis=0).tocsr()
    treatment = np.asarray(prepared["TREATED"])
    if (
        not np.isin(treatment, [0, 1]).all()
        or not (treatment == 0).any()
        or not (treatment == 1).any()
    ):
        raise ValueError("The cohort must contain both case (1) and control (0) rows.")
    return (
        matrix[treatment == 0],
        np.asarray(matrix[treatment == 1].mean(axis=0)).ravel(),
        prepared["Patient_token"][treatment == 0],
    )

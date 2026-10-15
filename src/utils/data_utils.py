"""Sparse data serialization and outcome-label helpers for the published analyses."""

import logging
import os
import pickle
from typing import Dict, List
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz, vstack


def insert_newline(text):
    words = text.split()
    new_text = ""
    for i, word in enumerate(words, start=1):
        new_text += word
        if i % 5 == 0:
            new_text += "\n"
        else:
            new_text += " "
    return new_text.strip()


def get_weights_at_loc(weights_path: str):
    if os.path.exists(weights_path):
        weights_dict = loadPickle(weights_path)
        return weights_dict["weights"], weights_dict["patient_tokens"]
    else:
        raise FileNotFoundError(weights_path)


def writePickle(obj, directory):
    logging.info("Writing to {}".format(directory))

    with open(directory, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadPickle(directory):
    logging.info("Reading from {}".format(directory))

    with open(directory, "rb") as handle:
        unserialized_data = pickle.load(handle)
        return unserialized_data


def save_data_dic(data, path):
    logging.info("Starting the saving process to {}...".format(path))

    if not os.path.exists(path):
        os.makedirs(path)

    # Save np.arrays
    np_path = os.path.join(path, "np_arrays")
    if not os.path.exists(np_path):
        os.makedirs(np_path)

    logging.info("Saving np.arrays...")
    for key in ["Patient_token", "DATESTAMP", "TREATED", "Patient_id"]:
        if key in data:
            np.save(os.path.join(np_path, key + ".npy"), data[key])

            logging.info(f"Saved {key}")
        else:
            logging.warning(f"{key} key not in data dic.")

    # Save lists
    list_path = os.path.join(path, "lists")
    if not os.path.exists(list_path):
        os.makedirs(list_path)

    logging.info("Saving lists...")
    for key in ["X_labels", "Y_labels"]:
        if key in data:
            with open(os.path.join(list_path, key + ".txt"), "w") as f:
                for item in data[key]:
                    f.write(f"{item}\n")

            logging.info(f"Saved {key}")
        else:
            logging.warning(f"{key} key not in data dic.")

    # Save CSR matrices in chunks
    csr_path = os.path.join(path, "csr_matrices")
    if not os.path.exists(csr_path):
        os.makedirs(csr_path)

    logging.info("Saving CSR matrices in chunks...")
    for key in ["X", "Y"]:
        if key in data:
            rows = data[key].shape[0]
            chunks = int(np.ceil(rows / 1_000_000))
            for chunk in range(chunks):
                start_idx = chunk * 1_000_000
                end_idx = min((chunk + 1) * 1_000_000, rows)
                chunk_path = os.path.join(csr_path, f"{key}_chunk_{chunk}.npz")
                save_npz(chunk_path, data[key][start_idx:end_idx])

                logging.info(f"Saved chunk {chunk + 1} of {chunks} for {key}")
        else:
            logging.warning(f"{key} key not in data dic.")

    logging.info("Saving process completed!")


def outcome_mappings(labels: List[str]):

    outcomes_list = [
        outcome[: outcome.index("_Day_")] if "_Day_" in outcome else None
        for outcome in labels
    ]
    outcome_to_indices: Dict[str, List[int]] = {}
    outcome_to_labels: Dict[str, List[str]] = {}

    # This takes care of outcomes that duplicated
    labels_processed = set()

    for i in range(len(outcomes_list)):
        if labels[i] in labels_processed:
            continue

        outcome = outcomes_list[i]
        if outcome is not None:
            if outcome not in outcome_to_indices:
                outcome_to_indices[outcome] = []
                outcome_to_labels[outcome] = []

            outcome_to_indices[outcome].append(i)
            outcome_to_labels[outcome].append(
                labels[i][labels[i].index(outcome) + len(outcome) + 1 :]
            )
            labels_processed.add(labels[i])

    return outcome_to_indices, outcome_to_labels


def extract_number(s):
    # Split by '_' and then take the part before '.npz' to get the number
    return int(s.split("_")[-1].split(".")[0])


def load_data_dic(path):
    """Load complete sparse arrays without altering the claim counts."""
    path = Path(path)
    data = {}
    for key in ["X", "Y"]:
        chunks = sorted(
            (path / "csr_matrices").glob(f"{key}_chunk_*.npz"),
            key=lambda p: extract_number(p.name),
        )
        if chunks:
            data[key] = vstack([load_npz(chunk) for chunk in chunks], format="csr")
    if "X" not in data:
        raise ValueError(f"No X matrix was found in {path}.")
    for key in ["Patient_token", "DATESTAMP", "TREATED", "Patient_id"]:
        array_path = path / "np_arrays" / f"{key}.npy"
        if array_path.exists():
            data[key] = np.load(array_path, allow_pickle=True)
            if len(data[key]) != data["X"].shape[0]:
                raise ValueError(f"Row count mismatch for {key} in {path}.")
    for key in ["X_labels", "Y_labels"]:
        list_path = path / "lists" / f"{key}.txt"
        if list_path.exists():
            data[key] = [
                line.strip()
                .replace(",", "")
                .replace("/", "")
                .replace("(", "")
                .replace(")", "")
                for line in list_path.read_text().splitlines()
            ]
            matrix_key = key[0]
            if matrix_key not in data or len(data[key]) != data[matrix_key].shape[1]:
                raise ValueError(f"Column label mismatch for {matrix_key} in {path}.")
    if "Y" in data and data["Y"].shape[0] != data["X"].shape[0]:
        raise ValueError(f"X and Y row counts differ in {path}.")
    return data

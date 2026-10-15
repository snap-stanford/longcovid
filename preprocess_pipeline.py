import argparse
import fcntl
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from src.utils.constants import (
    CENSUS_DATA_PATH,
    DIAGNOSES_TO_CODES_PROCESSED_PATH,
    PROCESSED_DEP_INDEX_PATH,
    RAW_DATA_BATCH_FOLDER,
    RAW_DATA_PATH,
)
from src.utils.data_utils import load_data_dic, save_data_dic

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Intermediate files created during pipeline
COHORT_FOLDERNAME = "cohort"
BATCH_RANGE_FILENAME = "batch_range.json"
BATCH_TO_FILES_FILENAME = "batch_files.json"
FINISHED_INDICATOR_FILE = "finished.txt"
ALL_OUTCOMES_FOLDER = "all_outcomes"
IN_PROGRESS_FILE = "in_progress.txt"
VISITS_FOLDER = "visits"
CLOSED_FOLDER = "closed"
STATIC_FOLDER = "static"
FINAL = "final"

# Constants relating to date -> month
FIRST_START_MONTH = -1440
LAST_START_MONTH = 690
NEGATIVE_MONTHS = [
    f"Day_N{abs(month)}_N{abs(month + 30)}" for month in range(FIRST_START_MONTH, 0, 30)
]
POSITIVE_MONTHS = ["Day_0_1", "Day_1_30"] + [
    f"Day_{month}_{month + 30}" for month in range(30, LAST_START_MONTH + 30, 30)
]
ALL_MONTHS = NEGATIVE_MONTHS + POSITIVE_MONTHS


# Categories of visits/static
VISIT_TO_CATEGORY = {
    "'\\N'": "Other_Visit",
    "'Outpatient Visit'": "Outpatient_Visit",
    "'Laboratory Visit'": "Laboratory_Visit",
    "'Home Visit'": "Home_Visit",
    "'Emergency Room Visit'": "Emergency_Room_Visit",
    "'Inpatient Visit'": "Inpatient_Visit",
    "'Non-hospital institution Visit'": "Non-hospital_institution_Visit",
    "'Ambulance Visit'": "Ambulance_Visit",
    "'Pharmacy Visit'": "Pharmacy Visit",
}
STATIC_FEATURES = ["GENDER_F", "GENDER_M", "GENDER_U", "Age", "ADI_NATRANK"]


def parse_icd_codes(value):
    """Read expanded, comma-separated identifiers from the final ICD table."""
    if pd.isna(value):
        return []
    return [code.strip() for code in str(value).split(",") if code.strip()]


def extract_numbers(filename) -> Tuple[int, int, int]:
    """Extract the numbers from the filename to use for sorting."""
    pattern = r"\d+_\d+_\d+"
    matches = re.findall(pattern, filename)

    if len(matches) != 1:
        raise Exception(f"Could not properly parse numbers from {filename}!")

    split = matches[0].split("_")
    return int(split[0]), int(split[1]), int(split[2])


def get_patient_id_range(
    data_path: str, cohort_filenames: List[str]
) -> Tuple[int, int]:
    """Get the max patient id and the minimum patient id from our cohort. Note: this assumes that after sorting,
    the lowest patient id is in the first file and highest is in the last file.
    """
    cohort_filenames_numbers = [
        (filename, extract_numbers(filename)) for filename in cohort_filenames
    ]
    cohort_filenames_numbers = sorted(
        cohort_filenames_numbers, key=lambda item: item[1]
    )

    # Check that patient ids are indeed sorted
    min_all = -1
    max_all = -1
    for filename, _ in cohort_filenames_numbers:
        df = pd.read_csv(
            os.path.join(data_path, filename),
            dtype={"PATIENT_ID": "uint32"},
            usecols=["PATIENT_ID"],
            delimiter="\t",
        )
        min_id = df["PATIENT_ID"].min()
        max_id = df["PATIENT_ID"].max()

        if max_all > 0:
            assert min_id == max_all + 1, (
                f"Cohorts unsorted between files! Current filename: {filename}"
            )

        comparison_array = np.arange(min_id, max_id + 1)
        assert np.array_equal(df["PATIENT_ID"].values, comparison_array), (
            f"Cohort table at {filename} is unsorted!"
        )

        if min_all < 0:
            min_all = min_id
        max_all = max_id

        logging.debug(f"FINISHED checking that {filename} is sorted.")

    logging.info(f"Min patient id: {min_all}, Max patient id: {max_all}.")
    return min_all, max_all


def write_batch_folders(
    num_batches: int, min_patient_id: int, max_patient_id: int, data_batch_path: str
):
    """Create the batch folders. Also, create a json file to keep track of the min and max
    patient id values to be processed in this batch.
    """
    # The last batch will have remainder elements
    batch_size = (max_patient_id - min_patient_id + 1) // (num_batches - 1)

    # Calculate min and max id for each batch
    batch_id_ranges = []
    for i in range(num_batches):
        batch_path = os.path.join(data_batch_path, str(i))
        os.makedirs(batch_path, exist_ok=True)

        if not os.path.exists(batch_path):
            os.chmod(batch_path, 0o777)

        start_id = min_patient_id + i * batch_size
        end_id = min(start_id + batch_size - 1, max_patient_id)
        batch_id_ranges.append((start_id, end_id))

        with open(os.path.join(batch_path, BATCH_RANGE_FILENAME), "w") as file:
            json.dump({"start_id": int(start_id), "end_id": int(end_id)}, file)

        if not os.path.exists(os.path.join(batch_path, BATCH_RANGE_FILENAME)):
            os.chmod(os.path.join(batch_path, BATCH_RANGE_FILENAME), 0o777)

    logging.info(f"FINISHED writing batch folders to {data_batch_path}.")


def map_batch_to_raw_files(filepath, batch_to_range):
    try:
        df = pd.read_csv(
            filepath,
            delimiter="\t",
            usecols=["PATIENT_ID"],
            dtype={"PATIENT_ID": "uint32"},
        )
    except Exception:
        df = pd.read_csv(
            filepath,
            delimiter="\t",
            usecols=["'PATIENT_ID'"],
            dtype={"'PATIENT_ID'": "uint32"},
        )
        df.rename(columns={"'PATIENT_ID'": "PATIENT_ID"}, inplace=True)

    # Check if minimum is within range or maximum is within range
    min_patient_id = df["PATIENT_ID"].min()
    max_patient_id = df["PATIENT_ID"].max()
    batch_to_filepath = {}
    for batch, batch_range in batch_to_range.items():
        if (
            batch_range["start_id"] <= max_patient_id
            and min_patient_id <= batch_range["end_id"]
        ):
            batch_to_filepath[batch] = filepath

    return batch_to_filepath


def handle_batch_to_raw_files_mappings(
    data_batch_path: str, table_filepaths: List[str], num_processes: int = 256
):
    batch_to_range = {}
    for batch in os.listdir(data_batch_path):
        with open(
            os.path.join(data_batch_path, batch, BATCH_RANGE_FILENAME), "r"
        ) as file:
            data = json.loads(file.read())

        batch_to_range[batch] = data

    args = []
    for filepath in table_filepaths:
        args.append((filepath, batch_to_range))

    logging.info("STARTING to map batch to raw data files.")
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(map_batch_to_raw_files, tqdm.tqdm(args, total=len(args)))

    combined_batch_to_fps: Dict[str, Set[str]] = {}
    for batch_to_fp in results:
        for batch, fp in batch_to_fp.items():
            if batch not in combined_batch_to_fps:
                combined_batch_to_fps[batch] = set()
            combined_batch_to_fps[batch].add(fp)

    for batch, fps in combined_batch_to_fps.items():
        with open(
            os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME), "w"
        ) as file:
            json.dump(list(fps), file)

        if not os.path.exists(
            os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME)
        ):
            os.chmod(
                os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME), 0o777
            )

    logging.info(f"FINISHED writing {BATCH_TO_FILES_FILENAME} files.")


def read_raw_tsv(filepath, usecols, dtype, parse_dates, delimiter="\t"):
    df = pd.read_csv(
        filepath,
        delimiter=delimiter,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
    )
    return df


def read_combine_batch_tsvs(filepaths, usecols, dtype, parse_dates, delimiter="\t"):
    """Read each export once and propagate worker errors to the caller."""
    if not filepaths:
        raise ValueError("No matching input export files were found.")

    def read_one(filepath):
        return read_raw_tsv(filepath, usecols, dtype, parse_dates, delimiter=delimiter)

    with ThreadPoolExecutor(max_workers=min(32, len(filepaths))) as executor:
        dataframes = list(executor.map(read_one, filepaths))
    return pd.concat(dataframes, ignore_index=True)


def day_to_month_category(x):
    if x == 0:
        return "Day_0_1"
    elif x > 0 and x < 30:
        return "Day_1_30"

    start = int(x // 30) * 30
    end = start + 30
    if x >= 0:
        return f"Day_{start}_{end}"
    else:
        if end == 0:
            return f"Day_N{abs(start)}_{end}"
        else:
            return f"Day_N{abs(start)}_N{abs(end)}"


def day_to_start_month(x):
    if x > 0 and x < 30:
        return 1

    return int(x // 30) * 30


def start_month_to_column_index(x):
    if x <= 0:
        return (x - FIRST_START_MONTH) // 30
    else:
        return ((x - FIRST_START_MONTH) // 30) + 1


def get_outcome_df(df_medical_headers, codes_list):
    """Match complete ICD identifiers from the final processed code table."""
    codes = {str(code).strip() for code in codes_list if str(code).strip()}
    if not codes:
        return None

    def matches(value):
        return any(str(code).strip() in codes for code in json.loads(value))

    return df_medical_headers[df_medical_headers["DIAGNOSIS_ARRAY"].apply(matches)]


def generate_features(
    data_batch_path, outcome, icd10_codes, icd9_codes, batch, verbose=True
):

    if os.path.exists(
        os.path.join(data_batch_path, batch, outcome, FINISHED_INDICATOR_FILE)
    ):
        return

    if not os.path.exists(os.path.join(data_batch_path, batch, outcome)):
        os.makedirs(os.path.join(data_batch_path, batch, outcome))

        if not os.path.exists(os.path.join(data_batch_path, batch, outcome)):
            os.chmod(os.path.join(data_batch_path, batch, outcome), 0o777)

    f = open(os.path.join(data_batch_path, batch, outcome, IN_PROGRESS_FILE), "ab")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.info(
            f"Another process has locked file: {os.path.join(data_batch_path, batch, outcome, IN_PROGRESS_FILE)}"
        )
        f.close()
        return

    # This batch hasn't been processed and nobody else is working on it
    with f:
        # Read in information about the batch
        with open(
            os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME), "r"
        ) as json_file:
            batch_raw_files = json.load(json_file)

        with open(
            os.path.join(data_batch_path, batch, BATCH_RANGE_FILENAME), "r"
        ) as json_file:
            batch_range = json.load(json_file)

        # Get the needed medical_headers files for the batch and filter it to only patients in this batch
        medical_headers_filepaths = [
            filepath for filepath in batch_raw_files if "medical_headers" in filepath
        ]
        cohort_filepaths = [
            filepath for filepath in batch_raw_files if "cohort" in filepath
        ]
        df_medical_headers = read_combine_batch_tsvs(
            medical_headers_filepaths,
            ["PATIENT_ID", "DAYS_SINCE_2012", "DIAGNOSIS_ARRAY", "ICD_9"],
            {"PATIENT_ID": "uint32", "DAYS_SINCE_2012": "uint16"},
            None,
        )
        df_medical_headers = df_medical_headers[
            df_medical_headers["DIAGNOSIS_ARRAY"] != "[]"
        ]

        logging.info(f"FINISHED reading in medical headers tsvs for {batch}/{outcome}.")
        df_medical_headers = df_medical_headers[
            (df_medical_headers["PATIENT_ID"] >= batch_range["start_id"])
            & (df_medical_headers["PATIENT_ID"] <= batch_range["end_id"])
        ]

        logging.info(
            f"DROPPING duplicate rows in medical headers. Length before de-duping: {len(df_medical_headers)}"
        )
        df_medical_headers.drop_duplicates(
            subset=["PATIENT_ID", "DAYS_SINCE_2012", "DIAGNOSIS_ARRAY"], inplace=True
        )
        logging.info(f"Length after de-duping: {len(df_medical_headers)}")

        # Get the needed cohort files for the batch and filter it to only patients in the batch. Convert incident
        # date to days since 2012
        df_cohort = read_combine_batch_tsvs(
            cohort_filepaths,
            ["PATIENT_ID", "CLAIM_DATE"],
            {"PATIENT_ID": "uint32"},
            ["CLAIM_DATE"],
        )

        logging.info(f"FINISHED reading in cohort tsvs for {batch}/{outcome}.")
        df_cohort = df_cohort[
            (df_cohort["PATIENT_ID"] >= batch_range["start_id"])
            & (df_cohort["PATIENT_ID"] <= batch_range["end_id"])
        ]
        df_cohort["days_since_20120101"] = (
            df_cohort["CLAIM_DATE"] - pd.to_datetime("2012-01-01")
        ).dt.days
        patient_to_incident_day = df_cohort.set_index("PATIENT_ID")[
            "days_since_20120101"
        ].to_dict()

        # Only get the medical_headers rows that have at least one of the diagnosis codes
        logging.info(f"FILTERING to outcome {outcome} for {batch}/{outcome}.")

        # First, split rows into icd9/icd10
        df_medical_headers_icd9 = df_medical_headers[df_medical_headers["ICD_9"] == "1"]
        df_medical_headers_icd10 = df_medical_headers[
            df_medical_headers["ICD_9"] == "0"
        ]
        df_medical_headers_both = df_medical_headers[
            df_medical_headers["ICD_9"] == "\\\\N"
        ]

        df_medical_headers_icd10_outcome = get_outcome_df(
            df_medical_headers_icd10, icd10_codes
        )
        df_medical_headers_icd9_outcome = get_outcome_df(
            df_medical_headers_icd9, icd9_codes
        )
        df_medical_headers_both_outcome = get_outcome_df(
            df_medical_headers_both, icd9_codes + icd10_codes
        )
        assert df_medical_headers_icd10_outcome is not None, (
            f"Problem in batch/outcome {batch}/{outcome} with icd10_codes: {icd10_codes}"
        )

        df_outcomes = [
            df
            for df in [
                df_medical_headers_icd10_outcome,
                df_medical_headers_icd9_outcome,
                df_medical_headers_both_outcome,
            ]
            if df is not None
        ]
        df_medical_headers_outcome = pd.concat(df_outcomes, ignore_index=True)

        # Group occurrences into month categories
        df_medical_headers_outcome["Days_to_incident"] = df_medical_headers_outcome[
            "PATIENT_ID"
        ].map(patient_to_incident_day)
        df_medical_headers_outcome["diagnosis_incident_date_diff"] = (
            df_medical_headers_outcome["DAYS_SINCE_2012"]
            - df_medical_headers_outcome["Days_to_incident"]
        )
        df_medical_headers_outcome["Month_category"] = df_medical_headers_outcome[
            "diagnosis_incident_date_diff"
        ].apply(lambda date_diff: day_to_month_category(date_diff))
        df_medical_headers_outcome["Start_month"] = df_medical_headers_outcome[
            "diagnosis_incident_date_diff"
        ].apply(lambda date_diff: day_to_start_month(date_diff))

        if verbose:
            logging.info(
                f"LENGTH of medical headers before grouping: {len(df_medical_headers_outcome)} for {batch}/{outcome}."
            )

        df_medical_headers_outcome = (
            df_medical_headers_outcome.groupby(
                ["PATIENT_ID", "Month_category", "Start_month"]
            )
            .size()
            .reset_index(name="Count")
        )

        if verbose:
            logging.info(
                f"LENGTH of medical headers after grouping: {len(df_medical_headers_outcome)} for {batch}/{outcome}."
            )

        # Filter to months that are only in the time period we care about
        df_medical_headers_outcome = df_medical_headers_outcome[
            (df_medical_headers_outcome["Start_month"] >= FIRST_START_MONTH)
            & (df_medical_headers_outcome["Start_month"] <= LAST_START_MONTH)
        ]

        # Turn the dataframe into a COO matrix, then convert to CSR
        df_medical_headers_outcome["column_index"] = df_medical_headers_outcome[
            "Start_month"
        ].apply(lambda month: start_month_to_column_index(month))
        df_medical_headers_outcome["row_index"] = (
            df_medical_headers_outcome["PATIENT_ID"] - batch_range["start_id"]
        )
        num_rows = batch_range["end_id"] - batch_range["start_id"] + 1
        coo_diagnosis = coo_matrix(
            (
                df_medical_headers_outcome["Count"],
                (
                    df_medical_headers_outcome["row_index"],
                    df_medical_headers_outcome["column_index"],
                ),
            ),
            shape=(num_rows, len(ALL_MONTHS)),
        )
        csr_diagnosis = coo_diagnosis.tocsr()

        # Save the data to a data dic
        patient_ids = np.arange(batch_range["start_id"], batch_range["end_id"] + 1)
        labels = [f"{outcome}_{month_label}" for month_label in ALL_MONTHS]
        batch_diagnosis_data_dict = {
            "X": csr_diagnosis,
            "Patient_id": patient_ids,
            "X_labels": labels,
        }
        save_data_dic(
            batch_diagnosis_data_dict, os.path.join(data_batch_path, batch, outcome)
        )
        # Write an empty text file to keep track of whether this finished
        with open(
            os.path.join(data_batch_path, batch, outcome, FINISHED_INDICATOR_FILE), "w"
        ):
            pass

        if not os.path.exists(
            os.path.join(data_batch_path, batch, outcome, FINISHED_INDICATOR_FILE)
        ):
            os.chmod(
                os.path.join(data_batch_path, batch, outcome, FINISHED_INDICATOR_FILE),
                0o777,
            )

        if verbose:
            logging.info(f"FINISHED writing data dic for {batch}/{outcome}.")


def process_medical_header_files(
    data_batch_path: str, num_batches: int = 256, num_processes: int = 500
):
    # Get all the diagnoses we will process
    df_diagnoses_to_categories = pd.read_csv(
        DIAGNOSES_TO_CODES_PROCESSED_PATH, dtype=str
    )
    df_diagnoses_to_categories["Diagnosis"] = df_diagnoses_to_categories[
        "Diagnosis"
    ].apply(lambda x: x.replace(" ", "_"))

    df_diagnoses_to_categories["Processed Codes ICD10"] = df_diagnoses_to_categories[
        "Processed Codes ICD10"
    ].apply(parse_icd_codes)
    df_diagnoses_to_categories["Processed Codes ICD9"] = df_diagnoses_to_categories[
        "Processed Codes ICD9"
    ].apply(parse_icd_codes)

    generate_features_args = []
    for _, row in df_diagnoses_to_categories.iterrows():
        outcome = row["Diagnosis"]

        for i in range(num_batches):
            batch = str(i)
            generate_features_args.append(
                (
                    data_batch_path,
                    outcome,
                    row["Processed Codes ICD10"],
                    row["Processed Codes ICD9"],
                    batch,
                    True,
                )
            )

    total_tasks = len(generate_features_args)
    logging.info(
        f"STARTING to map batch to raw data files. Will distribute {total_tasks} tasks across {num_processes} processes"
    )
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            generate_features, tqdm.tqdm(generate_features_args, total=total_tasks)
        )


def load_data_dics_threading(paths, num_threads: int = 50):
    """Load sparse batches with bounded workers and deterministic result order."""
    if not paths:
        raise ValueError("No sparse data paths were supplied.")
    with ThreadPoolExecutor(max_workers=min(num_threads, len(paths))) as executor:
        return list(executor.map(load_data_dic, paths))


def get_death_data_dic(df_died, batch_range):
    df_died["Month_category"] = df_died["death_relative_date"].apply(
        lambda date_diff: day_to_month_category(date_diff)
    )
    df_died["Start_month"] = df_died["death_relative_date"].apply(
        lambda date_diff: day_to_start_month(date_diff)
    )

    # Filter to months that are only in the time period we care about
    df_died = df_died[
        (df_died["Start_month"] >= FIRST_START_MONTH)
        & (df_died["Start_month"] <= LAST_START_MONTH)
    ]

    # Turn the dataframe into a COO matrix, then convert to CSR
    df_died["column_index"] = df_died["Start_month"].apply(
        lambda month: start_month_to_column_index(month)
    )
    df_died["row_index"] = df_died["PATIENT_ID"] - batch_range["start_id"]
    num_rows = batch_range["end_id"] - batch_range["start_id"] + 1
    coo_death = coo_matrix(
        (
            np.ones(len(df_died)),
            (
                df_died["row_index"],
                df_died["column_index"],
            ),
        ),
        shape=(num_rows, len(ALL_MONTHS)),
    )
    csr_death = coo_death.tocsr()
    patient_ids = np.arange(batch_range["start_id"], batch_range["end_id"] + 1)
    labels = [f"death_{month_label}" for month_label in ALL_MONTHS]
    batch_death_data_dict = {
        "X": csr_death,
        "Patient_id": patient_ids,
        "X_labels": labels,
    }
    return batch_death_data_dict


def combine_everything_in_batch(
    data_batch_path: str, batch: str, num_threads: int = 50
):

    # Get the cohort data so that the TREATMENT, Patient_token, and DATESTAMP columns
    # can be added
    with open(
        os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME), "r"
    ) as json_file:
        batch_raw_files = json.load(json_file)

    with open(
        os.path.join(data_batch_path, batch, BATCH_RANGE_FILENAME), "r"
    ) as json_file:
        batch_range = json.load(json_file)

    # Cohort dataframe
    cohort_filepaths = [
        filepath for filepath in batch_raw_files if "cohort" in filepath
    ]
    df_cohort = read_combine_batch_tsvs(
        cohort_filepaths,
        ["PATIENT_ID", "TREATMENT", "PATIENT_TOKEN_2", "CLAIM_DATE", "DATE_OF_DEATH"],
        {"PATIENT_ID": "uint32", "TREATMENT": "int8"},
        ["CLAIM_DATE"],
    )
    df_cohort = df_cohort[
        (df_cohort["PATIENT_ID"] >= batch_range["start_id"])
        & (df_cohort["PATIENT_ID"] <= batch_range["end_id"])
    ]
    df_cohort.sort_values(by=["PATIENT_ID"], inplace=True)

    # Get patients that died within 30 days of incident date so that we can remove them
    df_died = df_cohort[df_cohort["DATE_OF_DEATH"] != "\\\\N"]
    df_died["DATE_OF_DEATH"] = pd.to_datetime(df_died["DATE_OF_DEATH"])
    df_died["death_relative_date"] = (
        df_died["DATE_OF_DEATH"] - df_died["CLAIM_DATE"]
    ).dt.days

    # Get the data dic here since we are already handling death in this function
    data_dic_death = get_death_data_dic(df_died, batch_range)

    df_died_early = df_died[df_died["death_relative_date"] < 30]
    remove_patients = set(df_died_early["PATIENT_ID"])

    # Combine diagnoses
    df_diagnoses_to_categories = pd.read_csv(
        DIAGNOSES_TO_CODES_PROCESSED_PATH, dtype=str
    )
    df_diagnoses_to_categories["Diagnosis"] = df_diagnoses_to_categories[
        "Diagnosis"
    ].apply(lambda x: x.replace(" ", "_").replace("/", "_"))
    batch_outcomes_paths = [
        os.path.join(data_batch_path, batch, outcome)
        for outcome in df_diagnoses_to_categories["Diagnosis"]
    ]
    for path in batch_outcomes_paths:
        assert os.path.exists(os.path.join(path, FINISHED_INDICATOR_FILE)), (
            f"{os.path.join(path, FINISHED_INDICATOR_FILE)} doesn't exist!"
        )

    logging.info(f"START loading in outcome data dics for batch {batch}")
    data_dics = load_data_dics_threading(batch_outcomes_paths, num_threads=num_threads)

    # Sort data dics in ascending order of outcomes
    data_dics = sorted(data_dics, key=lambda x: x["X_labels"][0])

    logging.info(f"FINISHED loading in outcome data dics for batch {batch}")
    assert len(data_dics) == len(batch_outcomes_paths), (
        f"Something went wrong when loading in data dics for batch {batch}"
    )

    # First, check that everything is the same size and has the same values for patient ids
    for i in range(len(data_dics) - 1):
        assert data_dics[i]["X"].shape == data_dics[i + 1]["X"].shape
        assert np.array_equal(
            data_dics[i]["Patient_id"], data_dics[i + 1]["Patient_id"]
        )
        assert len(data_dics[i]["X_labels"]) == len(data_dics[i + 1]["X_labels"])

    # Add in closed events
    closed_path = os.path.join(data_batch_path, batch, CLOSED_FOLDER)
    assert os.path.exists(os.path.join(closed_path, FINISHED_INDICATOR_FILE)), (
        f"{os.path.join(closed_path, FINISHED_INDICATOR_FILE)} doesn't exist!"
    )

    data_dics.append(load_data_dic(closed_path))
    assert data_dics[-1]["X"].shape[0] == data_dics[-2]["X"].shape[0]
    assert data_dics[-1]["X"].shape[1] == data_dics[-2]["X"].shape[1]
    assert np.array_equal(data_dics[-1]["Patient_id"], data_dics[-2]["Patient_id"])
    assert len(data_dics[-1]["X_labels"]) == len(data_dics[-2]["X_labels"])

    # Add in visits
    visits_path = os.path.join(data_batch_path, batch, VISITS_FOLDER)
    assert os.path.exists(os.path.join(visits_path, FINISHED_INDICATOR_FILE)), (
        f"{os.path.join(visits_path, FINISHED_INDICATOR_FILE)} doesn't exist!"
    )

    data_dics.append(load_data_dic(visits_path))
    assert data_dics[-1]["X"].shape[0] == data_dics[-2]["X"].shape[0]
    assert data_dics[-1]["X"].shape[1] == data_dics[-2]["X"].shape[1] * len(
        VISIT_TO_CATEGORY
    )
    assert np.array_equal(data_dics[-1]["Patient_id"], data_dics[-2]["Patient_id"])
    assert len(data_dics[-1]["X_labels"]) == len(data_dics[-2]["X_labels"]) * len(
        VISIT_TO_CATEGORY
    )

    # Add in static
    static_path = os.path.join(data_batch_path, batch, STATIC_FOLDER)
    assert os.path.exists(os.path.join(static_path, FINISHED_INDICATOR_FILE)), (
        f"{os.path.join(static_path, FINISHED_INDICATOR_FILE)} doesn't exist!"
    )
    data_dics.append(load_data_dic(static_path))
    assert data_dics[-1]["X"].shape[0] == data_dics[-2]["X"].shape[0]
    assert data_dics[-1]["X"].shape[1] == len(STATIC_FEATURES)
    assert np.array_equal(data_dics[-1]["Patient_id"], data_dics[-2]["Patient_id"])
    assert len(data_dics[-1]["X_labels"]) == len(STATIC_FEATURES)

    # Add in death
    data_dics.append(data_dic_death)

    # Combine everything
    combined_x_labels = [
        element for data_dic in data_dics for element in data_dic["X_labels"]
    ]
    combined_patient_ids = data_dics[0]["Patient_id"]
    combined_X = hstack([data_dic["X"] for data_dic in data_dics])
    combined_data_dic = {
        "X": combined_X,
        "X_labels": combined_x_labels,
        "Patient_id": combined_patient_ids,
    }

    # Add in the cohort information
    assert np.array_equal(
        combined_data_dic["Patient_id"], df_cohort["PATIENT_ID"].values
    )
    combined_data_dic["DATESTAMP"] = df_cohort["CLAIM_DATE"].values
    combined_data_dic["TREATED"] = df_cohort["TREATMENT"].values
    combined_data_dic["Patient_token"] = df_cohort["PATIENT_TOKEN_2"].values

    # Lastly, remove patients that died within 30 days of incident date
    logging.info(f"REMOVING {len(remove_patients)} dead patients...")
    keep_row_indices = []
    for i in range(len(combined_data_dic["Patient_id"])):
        if combined_data_dic["Patient_id"][i] not in remove_patients:
            keep_row_indices.append(i)

    data_dic_death_removed = {
        "Patient_token": combined_data_dic["Patient_token"][keep_row_indices],
        "Patient_id": combined_data_dic["Patient_id"][keep_row_indices],
        "DATESTAMP": combined_data_dic["DATESTAMP"][keep_row_indices],
        "TREATED": combined_data_dic["TREATED"][keep_row_indices],
        "X_labels": combined_data_dic["X_labels"],
        "X": combined_data_dic["X"][keep_row_indices],
    }
    logging.info(
        "Data dic is now length {}.".format(
            len(data_dic_death_removed["Patient_token"])
        )
    )

    save_data_dic(
        data_dic_death_removed,
        os.path.join(data_batch_path, batch, ALL_OUTCOMES_FOLDER),
    )
    with open(
        os.path.join(
            data_batch_path, batch, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE
        ),
        "w",
    ):
        pass

    if not os.path.exists(
        os.path.join(
            data_batch_path, batch, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE
        )
    ):
        os.chmod(
            os.path.join(
                data_batch_path, batch, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE
            ),
            0o777,
        )


def create_visits_data_dic(data_batch_path: str, batch: str, num_threads: int = 50):
    with open(
        os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME), "r"
    ) as json_file:
        batch_raw_files = json.load(json_file)

    with open(
        os.path.join(data_batch_path, batch, BATCH_RANGE_FILENAME), "r"
    ) as json_file:
        batch_range = json.load(json_file)

    visit_filepaths = [
        filepath
        for filepath in batch_raw_files
        if "filtered_visit_types_with_visit_id" in filepath
    ]
    cohort_filepaths = [
        filepath for filepath in batch_raw_files if "cohort" in filepath
    ]

    # Read in and filter visits files
    df_visits = read_combine_batch_tsvs(
        visit_filepaths,
        [
            "'VISIT_ID_SIMPLE'",
            "'PATIENT_ID'",
            "'START_DAYS_SINCE_2012'",
            "'VISIT_TYPE'",
        ],
        {
            "'PATIENT_ID'": "uint32",
            "'START_DAYS_SINCE_2012'": "uint16",
            "'END_DAYS_SINCE_2012'": "uint16",
            "'VISIT_TYPE'": "category",
        },
        None,
    )
    df_visits.rename(
        columns={
            "'VISIT_ID_SIMPLE'": "VISIT_ID_SIMPLE",
            "'PATIENT_ID'": "PATIENT_ID",
            "'START_DAYS_SINCE_2012'": "START_DAYS_SINCE_2012",
            "'VISIT_TYPE'": "VISIT_TYPE",
        },
        inplace=True,
    )
    df_visits.drop_duplicates(
        subset=["VISIT_ID_SIMPLE", "PATIENT_ID", "START_DAYS_SINCE_2012", "VISIT_TYPE"],
        keep="first",
        inplace=True,
        ignore_index=True,
    )

    unique_visit_types = set(df_visits["VISIT_TYPE"])
    assert unique_visit_types == set(VISIT_TO_CATEGORY.keys()), (
        f"Unique visit types for batch {batch} are not expected; got {unique_visit_types}"
    )
    df_visits["VISIT_TYPE"].replace(VISIT_TO_CATEGORY, inplace=True)
    df_visits = df_visits[
        (df_visits["PATIENT_ID"] >= batch_range["start_id"])
        & (df_visits["PATIENT_ID"] <= batch_range["end_id"])
    ]
    logging.info(f"FINISHED loading and processing visits table for batch {batch}.")

    # Read in and process cohort files
    df_cohort = read_combine_batch_tsvs(
        cohort_filepaths,
        ["PATIENT_ID", "CLAIM_DATE"],
        {"PATIENT_ID": "uint32"},
        ["CLAIM_DATE"],
    )

    df_cohort = df_cohort[
        (df_cohort["PATIENT_ID"] >= batch_range["start_id"])
        & (df_cohort["PATIENT_ID"] <= batch_range["end_id"])
    ]
    df_cohort["days_since_20120101"] = (
        df_cohort["CLAIM_DATE"] - pd.to_datetime("2012-01-01")
    ).dt.days
    patient_to_incident_day = df_cohort.set_index("PATIENT_ID")[
        "days_since_20120101"
    ].to_dict()

    logging.info(f"FINISHED loading and processing cohort table for batch {batch}.")

    # Group occurrences into month categories
    df_visits["Days_to_incident"] = df_visits["PATIENT_ID"].map(patient_to_incident_day)
    df_visits["diagnosis_incident_date_diff"] = (
        df_visits["START_DAYS_SINCE_2012"] - df_visits["Days_to_incident"]
    )
    df_visits["Month_category"] = df_visits["diagnosis_incident_date_diff"].apply(
        lambda date_diff: day_to_month_category(date_diff)
    )
    df_visits["Start_month"] = df_visits["diagnosis_incident_date_diff"].apply(
        lambda date_diff: day_to_start_month(date_diff)
    )

    df_visits = df_visits[
        (df_visits["Start_month"] >= FIRST_START_MONTH)
        & (df_visits["Start_month"] <= LAST_START_MONTH)
    ]

    # Do this so that the groupby will go in the same order for all batches
    df_visits.sort_values(by=["VISIT_TYPE"], inplace=True)

    logging.info(f"STARTING conversions to sparse visit matrices for {batch}.")
    labels = []
    Xs = []
    for visit_type, df_visits_type in df_visits.groupby("VISIT_TYPE"):
        df_visits_type = (
            df_visits_type.groupby(["PATIENT_ID", "Month_category", "Start_month"])
            .size()
            .reset_index(name="Count")
        )
        visit_type = visit_type.replace("'", "")

        # Turn the dataframe into a COO matrix, then convert to CSR
        df_visits_type["column_index"] = df_visits_type["Start_month"].apply(
            lambda month: start_month_to_column_index(month)
        )
        df_visits_type["row_index"] = (
            df_visits_type["PATIENT_ID"] - batch_range["start_id"]
        )
        num_rows = batch_range["end_id"] - batch_range["start_id"] + 1
        coo_visit = coo_matrix(
            (
                df_visits_type["Count"],
                (
                    df_visits_type["row_index"],
                    df_visits_type["column_index"],
                ),
            ),
            shape=(num_rows, len(ALL_MONTHS)),
        )
        csr_visit = coo_visit.tocsr()

        Xs.append(csr_visit)
        labels.extend([f"{visit_type}_{month_label}" for month_label in ALL_MONTHS])
        logging.info(f"FINISHED visit type {visit_type} for {batch}.")

    combined_X = hstack(Xs)
    combined_data_dic = {
        "X": combined_X,
        "X_labels": labels,
        "Patient_id": np.arange(batch_range["start_id"], batch_range["end_id"] + 1),
    }
    logging.info(f"FINISHED combining visit data dics for {batch}.")

    save_data_dic(
        combined_data_dic, os.path.join(data_batch_path, batch, VISITS_FOLDER)
    )
    with open(
        os.path.join(data_batch_path, batch, VISITS_FOLDER, FINISHED_INDICATOR_FILE),
        "w",
    ):
        pass

    if not os.path.exists(
        os.path.join(data_batch_path, batch, VISITS_FOLDER, FINISHED_INDICATOR_FILE)
    ):
        os.chmod(
            os.path.join(
                data_batch_path, batch, VISITS_FOLDER, FINISHED_INDICATOR_FILE
            ),
            0o777,
        )

    logging.info(f"FINISHED processing visits for batch {batch}.")


def create_batch_visits_data_dics(
    data_batch_path: str,
    num_batches: int = 256,
    num_threads: int = 50,
    num_processes: int = 256,
):
    process_visits_args = []
    skipped = []
    for i in range(num_batches):
        batch = str(i)

        if os.path.exists(
            os.path.join(data_batch_path, batch, VISITS_FOLDER, FINISHED_INDICATOR_FILE)
        ):
            skipped.append(batch)
        else:
            process_visits_args.append((data_batch_path, batch, num_threads))

    if len(process_visits_args) == 0:
        logging.info("Visits data dics exist! SKIPPING creation.")
        return

    if len(skipped) > 0:
        logging.info(
            f"SKIPPING creation for the {len(skipped)} visits data dics as they already exist."
        )

    total_tasks = len(process_visits_args)
    logging.info("STARTING to create visits batch data dic.")
    with Pool(processes=min(total_tasks, num_processes)) as pool:
        pool.starmap(
            create_visits_data_dic,
            tqdm.tqdm(process_visits_args, total=total_tasks),
        )


def create_closed_data_dic(data_batch_path: str, batch: str, num_threads: int = 50):
    with open(
        os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME), "r"
    ) as json_file:
        batch_raw_files = json.load(json_file)

    with open(
        os.path.join(data_batch_path, batch, BATCH_RANGE_FILENAME), "r"
    ) as json_file:
        batch_range = json.load(json_file)

    closed_filepaths = [
        filepath for filepath in batch_raw_files if "closed_events" in filepath
    ]

    cohort_filepaths = [
        filepath for filepath in batch_raw_files if "cohort" in filepath
    ]

    # Read in and filter visits files
    df_closed = read_combine_batch_tsvs(
        closed_filepaths,
        [
            "PATIENT_ID",
            "DAYS_SINCE_2012",
        ],
        {
            "PATIENT_ID": "uint32",
            "DAYS_SINCE_2012": "uint16",
        },
        None,
    )
    df_closed.drop_duplicates(
        subset=["PATIENT_ID", "DAYS_SINCE_2012"],
        keep="first",
        inplace=True,
        ignore_index=True,
    )

    df_closed = df_closed[
        (df_closed["PATIENT_ID"] >= batch_range["start_id"])
        & (df_closed["PATIENT_ID"] <= batch_range["end_id"])
    ]
    logging.info(f"FINISHED loading and processing visits table for batch {batch}.")

    # Read in and process cohort files
    df_cohort = read_combine_batch_tsvs(
        cohort_filepaths,
        ["PATIENT_ID", "CLAIM_DATE"],
        {"PATIENT_ID": "uint32"},
        ["CLAIM_DATE"],
    )

    df_cohort = df_cohort[
        (df_cohort["PATIENT_ID"] >= batch_range["start_id"])
        & (df_cohort["PATIENT_ID"] <= batch_range["end_id"])
    ]
    df_cohort["days_since_20120101"] = (
        df_cohort["CLAIM_DATE"] - pd.to_datetime("2012-01-01")
    ).dt.days
    patient_to_incident_day = df_cohort.set_index("PATIENT_ID")[
        "days_since_20120101"
    ].to_dict()

    logging.info(f"FINISHED loading and processing cohort table for batch {batch}.")

    # Group occurrences into month categories
    df_closed["Days_to_incident"] = df_closed["PATIENT_ID"].map(patient_to_incident_day)
    df_closed["diagnosis_incident_date_diff"] = (
        df_closed["DAYS_SINCE_2012"] - df_closed["Days_to_incident"]
    )
    df_closed["Month_category"] = df_closed["diagnosis_incident_date_diff"].apply(
        lambda date_diff: day_to_month_category(date_diff)
    )
    df_closed["Start_month"] = df_closed["diagnosis_incident_date_diff"].apply(
        lambda date_diff: day_to_start_month(date_diff)
    )

    df_closed = df_closed[
        (df_closed["Start_month"] >= FIRST_START_MONTH)
        & (df_closed["Start_month"] <= LAST_START_MONTH)
    ]

    logging.info(f"STARTING conversions to sparse closed matrices for {batch}.")

    # Turn the dataframe into a COO matrix, then convert to CSR

    df_closed = (
        df_closed.groupby(["PATIENT_ID", "Month_category", "Start_month"])
        .size()
        .reset_index(name="Count")
    )

    df_closed["column_index"] = df_closed["Start_month"].apply(
        lambda month: start_month_to_column_index(month)
    )
    df_closed["row_index"] = df_closed["PATIENT_ID"] - batch_range["start_id"]
    num_rows = batch_range["end_id"] - batch_range["start_id"] + 1
    coo_closed = coo_matrix(
        (
            np.ones(len(df_closed)),
            (
                df_closed["row_index"],
                df_closed["column_index"],
            ),
        ),
        shape=(num_rows, len(ALL_MONTHS)),
    )
    csr_closed = coo_closed.tocsr()
    logging.info(f"FINISHED csr matrix of closed_events for {batch}.")

    data_dic = {
        "X": csr_closed,
        "X_labels": [f"closed_{month_label}" for month_label in ALL_MONTHS],
        "Patient_id": np.arange(batch_range["start_id"], batch_range["end_id"] + 1),
    }

    save_data_dic(data_dic, os.path.join(data_batch_path, batch, CLOSED_FOLDER))
    with open(
        os.path.join(data_batch_path, batch, CLOSED_FOLDER, FINISHED_INDICATOR_FILE),
        "w",
    ):
        pass

    if not os.path.exists(
        os.path.join(data_batch_path, batch, CLOSED_FOLDER, FINISHED_INDICATOR_FILE)
    ):
        os.chmod(
            os.path.join(
                data_batch_path, batch, CLOSED_FOLDER, FINISHED_INDICATOR_FILE
            ),
            0o777,
        )

    logging.info(f"FINISHED processing visits for batch {batch}.")


def create_batch_closed_data_dics(
    data_batch_path: str,
    num_batches: int = 256,
    num_threads: int = 50,
    num_processes: int = 256,
):
    process_closed_args = []
    skipped = []
    for i in range(num_batches):
        batch = str(i)

        if os.path.exists(
            os.path.join(data_batch_path, batch, CLOSED_FOLDER, FINISHED_INDICATOR_FILE)
        ):
            skipped.append(batch)
        else:
            process_closed_args.append((data_batch_path, batch, num_threads))

    if len(process_closed_args) == 0:
        logging.info("Closed data dics exist! SKIPPING creation.")
        return

    if len(skipped) > 0:
        logging.info(
            f"SKIPPING creation for the {len(skipped)} closed data dics as they already exist."
        )

    total_tasks = len(process_closed_args)
    logging.info("STARTING to create closed batch data dic.")
    with Pool(processes=min(total_tasks, num_processes)) as pool:
        pool.starmap(
            create_closed_data_dic,
            tqdm.tqdm(process_closed_args, total=total_tasks),
        )


def create_static_data_dic(
    data_batch_path: str,
    batch,
    num_threads: int = 50,
):
    with open(
        os.path.join(data_batch_path, batch, BATCH_TO_FILES_FILENAME), "r"
    ) as json_file:
        batch_raw_files = json.load(json_file)

    with open(
        os.path.join(data_batch_path, batch, BATCH_RANGE_FILENAME), "r"
    ) as json_file:
        batch_range = json.load(json_file)

    df_state = pd.read_csv(PROCESSED_DEP_INDEX_PATH)
    df_state["ZIP_3"] = df_state["ZIP_3"].astype(str)

    # Again, bring back leading 0s.
    df_state["ZIP_3"] = df_state["ZIP_3"].str.zfill(3)

    zip3_to_adi_natrank = df_state.set_index("ZIP_3")["ADI_NATRANK"].to_dict()

    static_filepaths = [
        filepath for filepath in batch_raw_files if "static_covariates" in filepath
    ]
    cohort_filepaths = [
        filepath for filepath in batch_raw_files if "cohort" in filepath
    ]

    # Read in static df
    gender_categories = {"F", "M", "U"}
    df_static = read_combine_batch_tsvs(
        static_filepaths,
        ["PATIENT_ID", "PATIENT_DOB", "PATIENT_GENDER", "PATIENT_ZIP3"],
        {"PATIENT_ID": "uint32", "PATIENT_GENDER": "category"},
        ["PATIENT_DOB"],
    )
    assert set(df_static["PATIENT_GENDER"]) == gender_categories
    df_static = df_static[
        (df_static["PATIENT_ID"] >= batch_range["start_id"])
        & (df_static["PATIENT_ID"] <= batch_range["end_id"])
    ]

    # Read in cohort df
    df_cohort = read_combine_batch_tsvs(
        cohort_filepaths,
        ["PATIENT_ID", "CLAIM_DATE"],
        {"PATIENT_ID": "uint32"},
        ["CLAIM_DATE"],
    )

    df_cohort = df_cohort[
        (df_cohort["PATIENT_ID"] >= batch_range["start_id"])
        & (df_cohort["PATIENT_ID"] <= batch_range["end_id"])
    ]
    patient_to_incident_day = df_cohort.set_index("PATIENT_ID")["CLAIM_DATE"].to_dict()

    # Handle age
    df_static["CLAIM_DATE"] = df_static["PATIENT_ID"].map(patient_to_incident_day)
    df_static["Age"] = (
        df_static["CLAIM_DATE"] - df_static["PATIENT_DOB"]
    ).dt.days // 365

    # Handle gender
    df_gender = pd.get_dummies(
        df_static["PATIENT_GENDER"], prefix="GENDER", dtype="int8"
    )
    df_static = pd.concat([df_static, df_gender], axis=1)

    # Handle deprivation index
    df_static["PATIENT_ZIP3"] = df_static["PATIENT_ZIP3"].astype(str)
    df_static["ADI_NATRANK"] = df_static["PATIENT_ZIP3"].map(zip3_to_adi_natrank)
    df_static = df_static.groupby("PATIENT_ID", as_index=False).agg(
        {
            "GENDER_F": "first",
            "GENDER_M": "first",
            "GENDER_U": "first",
            "Age": "first",
            "ADI_NATRANK": "mean",
        }
    )
    df_static.sort_values(by="PATIENT_ID", inplace=True)
    assert np.array_equal(
        df_static["PATIENT_ID"].values,
        np.arange(batch_range["start_id"], batch_range["end_id"] + 1),
    ), f"df_static problem for batch {batch}"

    sparse_matrix = csr_matrix(df_static[STATIC_FEATURES].values)
    data_dic = {
        "X": sparse_matrix,
        "X_labels": STATIC_FEATURES,
        "Patient_id": df_static["PATIENT_ID"].values,
    }

    save_data_dic(data_dic, os.path.join(data_batch_path, batch, STATIC_FOLDER))
    with open(
        os.path.join(data_batch_path, batch, STATIC_FOLDER, FINISHED_INDICATOR_FILE),
        "w",
    ):
        pass

    if not os.path.exists(
        os.path.join(data_batch_path, batch, STATIC_FOLDER, FINISHED_INDICATOR_FILE)
    ):
        os.chmod(
            os.path.join(
                data_batch_path, batch, STATIC_FOLDER, FINISHED_INDICATOR_FILE
            ),
            0o777,
        )


def create_batch_static_data_dics(
    data_batch_path: str,
    num_batches: int = 256,
    num_threads: int = 50,
    num_processes: int = 256,
):
    process_static_args = []
    skipped = []
    for i in range(num_batches):
        batch = str(i)

        if os.path.exists(
            os.path.join(data_batch_path, batch, STATIC_FOLDER, FINISHED_INDICATOR_FILE)
        ):
            skipped.append(batch)
        else:
            process_static_args.append((data_batch_path, batch, num_threads))

    if len(process_static_args) == 0:
        logging.info("static data dics exist! SKIPPING creation.")
        return

    if len(skipped) > 0:
        logging.info(
            f"SKIPPING creation for the {len(skipped)} static data dics as they already exist."
        )

    total_tasks = len(process_static_args)
    logging.info("STARTING to create visits batch data dic.")
    with Pool(processes=min(total_tasks, num_processes)) as pool:
        pool.starmap(
            create_static_data_dic,
            tqdm.tqdm(process_static_args, total=total_tasks),
        )


def create_batch_data_dics(
    data_batch_path: str,
    num_batches: int = 256,
    num_threads: int = 50,
    num_processes: int = 256,
):
    combine_everything_args = []
    skipped = []

    for i in range(num_batches):
        batch = str(i)

        if os.path.exists(
            os.path.join(
                data_batch_path, batch, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE
            )
        ):
            skipped.append(batch)
        else:
            combine_everything_args.append((data_batch_path, batch, num_threads))

    if len(combine_everything_args) == 0:
        logging.info("Batch data dics exist! SKIPPING combining.")
        return

    if len(skipped) > 0:
        logging.info(
            f"SKIPPING combining for the {len(skipped)} batches as they already exist."
        )

    total_tasks = len(combine_everything_args)
    logging.info("STARTING to combine outcomes into batch data dic.")

    with Pool(processes=min(total_tasks, num_processes)) as pool:
        pool.starmap(
            combine_everything_in_batch,
            tqdm.tqdm(combine_everything_args, total=total_tasks),
        )


def filter_alphanumeric(s):
    if "Day" not in s:
        return ("00000" + s, 0)
    else:
        # Split the string into parts before and after '_Day_'
        before_day, after_day = s.split("_Day_")

        # Extract the part that looks like '_N30_' where N represents a negative number
        negative_number_part = after_day.split("_")[0]
        if "N" in negative_number_part:
            # Get the integer value of the negative number part
            negative_number = int(negative_number_part[1:])
            return (before_day, -negative_number)
        else:
            positive_number = int(negative_number_part)
            return (before_day, positive_number)


def reorder_columns(data_dic):
    # Get the filtered alphanumeric column names
    filtered_labels = [filter_alphanumeric(label) for label in data_dic["X_labels"]]

    # Pair each filtered label with its original index
    indexed_labels = list(enumerate(filtered_labels))

    # Sort the pairs by label, alphanumeric sorting
    sorted_indexed_labels = sorted(indexed_labels, key=lambda x: x[1])

    # Extract the sorted indices
    sorted_indices = [index for index, _ in sorted_indexed_labels]

    # Reorder the CSR matrix columns using the sorted indices
    data_dic["X"] = data_dic["X"][:, sorted_indices]

    # Also reorder the labels to match the new column order
    data_dic["X_labels"] = [data_dic["X_labels"][i] for i in sorted_indices]


def combine_all_batch_data_dics(
    data_batch_path: str, num_batches: int = 256, num_threads: int = 50
):
    if os.path.exists(
        os.path.join(data_batch_path, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE)
    ):
        logging.info("Combined data dic already exists. SKIPPING creation...")
        return

    batch_data_dic_paths = []
    for i in range(num_batches):
        if not os.path.exists(
            os.path.join(
                data_batch_path, str(i), ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE
            )
        ):
            raise Exception(
                f"Cannot combine all batches because batch {i} is not finished!"
            )

        batch_data_dic_paths.append(
            os.path.join(data_batch_path, str(i), ALL_OUTCOMES_FOLDER)
        )

    logging.info("START loading in batch data dics.")
    data_dics = load_data_dics_threading(batch_data_dic_paths, num_threads=num_threads)

    for i in range(len(data_dics)):
        logging.critical("Reordering column for batch {}".format(i))
        reorder_columns(data_dics[i])

    # Verify that shapes of X matrices are compatible
    for i in range(len(data_dics) - 1):
        logging.critical("Checking for mixups {}".format(i))

        if [filter_alphanumeric(x) for x in data_dics[i]["X_labels"]] != [
            filter_alphanumeric(x) for x in data_dics[i + 1]["X_labels"]
        ]:
            logging.critical(
                "Sorting failed. Batch {} and {} incompatible AFTER SORTING, sizes {} and {}".format(
                    i,
                    i + 1,
                    len(data_dics[i]["X_labels"]),
                    len(data_dics[i + 1]["X_labels"]),
                )
            )

        assert data_dics[i]["X"].shape[1] == data_dics[i + 1]["X"].shape[1], (
            "Mismatch in shape for different batches!"
        )
        assert [filter_alphanumeric(x) for x in data_dics[i]["X_labels"]] == [
            filter_alphanumeric(x) for x in data_dics[i + 1]["X_labels"]
        ]

    # Combine the data dics into a single one by stacking them
    combined_X = vstack([data_dic["X"] for data_dic in data_dics])
    combined_x_labels = data_dics[0]["X_labels"]
    combined_patient_ids = [
        element for data_dic in data_dics for element in data_dic["Patient_id"]
    ]
    combined_patient_token = [
        element for data_dic in data_dics for element in data_dic["Patient_token"]
    ]
    combined_datestamp = [
        element for data_dic in data_dics for element in data_dic["DATESTAMP"]
    ]
    combined_treated = [
        element for data_dic in data_dics for element in data_dic["TREATED"]
    ]
    combined_data_dic = {
        "X": combined_X,
        "X_labels": combined_x_labels,
        "Patient_id": combined_patient_ids,
        "TREATED": combined_treated,
        "DATESTAMP": combined_datestamp,
        "Patient_token": combined_patient_token,
    }
    save_data_dic(combined_data_dic, os.path.join(data_batch_path, ALL_OUTCOMES_FOLDER))
    with open(
        os.path.join(data_batch_path, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE),
        "w",
    ):
        pass

    if not os.path.exists(
        os.path.join(data_batch_path, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE)
    ):
        os.chmod(
            os.path.join(data_batch_path, ALL_OUTCOMES_FOLDER, FINISHED_INDICATOR_FILE),
            0o777,
        )

    logging.info("FINISHED combining data_dics across batches.")


def apply_final_processing(
    final_output_location: str, data_batch_path: str, patient_tokens_file: str
):
    """Select the verified enrollment rows and separate pre- and post-enrollment columns."""
    membership = pd.read_csv(patient_tokens_file)
    if (
        membership["PATIENT_ID"].duplicated().any()
        or not membership["TREATMENT"].isin([0, 1]).all()
    ):
        raise ValueError(
            "Membership requires unique PATIENT_ID values and binary TREATMENT values."
        )
    outcomes_path = os.path.join(data_batch_path, ALL_OUTCOMES_FOLDER)
    if not os.path.exists(os.path.join(outcomes_path, FINISHED_INDICATOR_FILE)):
        raise ValueError("Complete all outcome batches before final cohort selection.")
    data = load_data_dic(outcomes_path)
    treatment_by_id = membership.set_index("PATIENT_ID")["TREATMENT"].to_dict()
    rows = np.array(
        [
            i
            for i, patient_id in enumerate(data["Patient_id"])
            if patient_id in treatment_by_id
        ],
        dtype=int,
    )
    if set(membership["PATIENT_ID"]) - set(data["Patient_id"][rows]):
        raise ValueError(
            "Some membership IDs are absent from the completed sparse batches."
        )
    x_columns = [
        i
        for i, label in enumerate(data["X_labels"])
        if "_Day_" not in label or "_Day_N" in label
    ]
    y_columns = [i for i in range(len(data["X_labels"])) if i not in set(x_columns)]
    final = {
        key: data[key][rows] for key in ["Patient_token", "Patient_id", "DATESTAMP"]
    }
    final.update(
        TREATED=np.array(
            [treatment_by_id[patient_id] for patient_id in final["Patient_id"]]
        ),
        X=data["X"][rows][:, x_columns],
        Y=data["X"][rows][:, y_columns],
        X_labels=np.array(data["X_labels"])[x_columns].tolist(),
        Y_labels=np.array(data["X_labels"])[y_columns].tolist(),
    )
    output = os.path.join(data_batch_path, final_output_location)
    save_data_dic(final, output)
    return output


def get_patient_census_df(df_static_cohort):
    census_cols = ["Mean Income", "white_percentage", "no bachelor degree"]
    df_census = pd.read_csv(
        os.path.join(CENSUS_DATA_PATH, "processed_census_data.csv"), dtype={"ZIP3": str}
    )
    df_census["no bachelor degree"] = (
        1
        - df_census[
            "Estimate!!Total!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher Percentage"
        ]
    )
    df_census = df_census[["ZIP3"] + census_cols]
    df_census = df_census.set_index("ZIP3")

    for col in tqdm.tqdm(census_cols):
        logging.info(f"Processing census column: {col}")
        census_dict = df_census[col].to_dict()
        df_static_cohort[col] = df_static_cohort["PATIENT_ZIP3"].apply(
            lambda zipcode: map_census_data(zipcode, census_dict)
        )

    df_static_cohort = df_static_cohort[["PATIENT_ID"] + census_cols]
    df_patients_grouped = df_static_cohort.groupby("PATIENT_ID").mean().reset_index()

    logging.info("PRINTING length of patient dataframe and number of nan values")
    logging.info("Total Patients: {}".format(len(df_patients_grouped)))
    logging.info(
        "Missing Mean Income: {}".format(
            len(df_patients_grouped[df_patients_grouped["Mean Income"].isna()])
        )
    )
    logging.info(
        "Missing Race: {}".format(
            len(df_patients_grouped[df_patients_grouped["white_percentage"].isna()])
        )
    )
    logging.info(
        "Missing Education: {}".format(
            len(df_patients_grouped[df_patients_grouped["no bachelor degree"].isna()])
        )
    )

    return df_patients_grouped


def map_census_data(zipcode, census_dict):
    return census_dict.get(zipcode, np.nan)


def add_census_features(data_dic_location: str, static_filepaths: List[str]):
    data_dic = load_data_dic(data_dic_location)
    output_path = data_dic_location + "_census"

    # Map the patient ids to census data
    df_static_cohort = read_combine_batch_tsvs(
        static_filepaths,
        ["PATIENT_ID", "PATIENT_ZIP3"],
        {"PATIENT_ID": "uint32", "PATIENT_ZIP3": str},
        None,
        delimiter="\t",
    )
    df_static_cohort = df_static_cohort[
        df_static_cohort["PATIENT_ID"].isin(set(data_dic["Patient_id"]))
    ]
    df_patients_grouped = get_patient_census_df(df_static_cohort)

    # Order the dataframe to match the data_dic order and fill NaN values
    id_to_position = {
        patient_id: position
        for position, patient_id in enumerate(data_dic["Patient_id"])
    }
    df_patients_grouped["sort_index"] = df_patients_grouped["PATIENT_ID"].map(
        id_to_position
    )
    df_patients_grouped_sorted = df_patients_grouped.sort_values("sort_index").drop(
        columns="sort_index"
    )
    df_patients_grouped_sorted["Mean Income"].fillna(
        df_patients_grouped["Mean Income"].mean(), inplace=True
    )
    df_patients_grouped_sorted["white_percentage"].fillna(
        df_patients_grouped["white_percentage"].mean(), inplace=True
    )
    df_patients_grouped_sorted["no bachelor degree"].fillna(
        df_patients_grouped["no bachelor degree"].mean(), inplace=True
    )

    # Add the census columns to the data_dic
    df_patients_grouped_sorted["nonwhite_or_hispanic"] = (
        1 - df_patients_grouped_sorted["white_percentage"]
    )
    census_labels = ["Mean Income", "nonwhite_or_hispanic", "no bachelor degree"]
    census_matrix = csr_matrix(df_patients_grouped_sorted[census_labels].values)

    data_dic["X"] = hstack([data_dic["X"], census_matrix])
    data_dic["X_labels"] = data_dic["X_labels"] + census_labels

    save_data_dic(data_dic, output_path)
    logging.info(f"wrote final data dic with census data to {output_path}")


def main(
    num_batches: int = 256,
    num_processes: int = min(16, os.cpu_count() or 1),
    multiple_servers: bool = False,
    patient_tokens_file: Optional[str] = None,
    final_output_location: str = "FINAL",
    cohort_folder_name: str = COHORT_FOLDERNAME,
    raw_data_batch_folder: str = RAW_DATA_BATCH_FOLDER,
    only_census_step: bool = False,
    prepare_only: bool = False,
):

    data_path = RAW_DATA_PATH
    data_batch_path = os.path.join(data_path, raw_data_batch_folder)
    logging.info(f"USING data at path {data_path}.")

    # Get lists of relevant filenames/filepaths
    medical_header_filenames = [
        filename
        for filename in os.listdir(
            os.path.join(data_path, "medical_headers_with_visit_id")
        )
        if "medical_headers_with_id_.tsv" in filename
    ]
    medical_header_filepaths = [
        os.path.join(data_path, "medical_headers_with_visit_id", filename)
        for filename in medical_header_filenames
    ]

    static_filenames = [
        filename
        for filename in os.listdir(os.path.join(data_path, "static_covariates"))
        if "static_.tsv" in filename
    ]
    static_filepaths = [
        os.path.join(data_path, "static_covariates", filename)
        for filename in static_filenames
    ]

    cohort_filenames = [
        filename
        for filename in os.listdir(os.path.join(data_path, cohort_folder_name))
        if "cohort_.tsv" in filename
    ]

    logging.info("Using %d cohort export files.", len(cohort_filenames))

    cohort_filepaths = [
        os.path.join(data_path, cohort_folder_name, filename)
        for filename in cohort_filenames
    ]

    filtered_visit_filenames = [
        filename
        for filename in os.listdir(
            os.path.join(data_path, "filtered_visit_types_with_visit_id")
        )
        if "filtered_visit_types_with_id_.tsv" in filename
    ]
    filtered_visit_filepaths = [
        os.path.join(data_path, "filtered_visit_types_with_visit_id", filename)
        for filename in filtered_visit_filenames
    ]

    closed_filenames = [
        filename
        for filename in os.listdir(os.path.join(data_path, "closed_events"))
        if "closed_events_.tsv" in filename
    ]
    closed_filepaths = [
        os.path.join(data_path, "closed_events", filename)
        for filename in closed_filenames
    ]

    if only_census_step:
        processed_data_dic_path = os.path.join(data_batch_path, final_output_location)

        if not os.path.exists(processed_data_dic_path):
            raise Exception(
                f"Processed data_dic path {processed_data_dic_path} does not exist. Cannot use only_census_step option."
            )

        add_census_features(processed_data_dic_path, static_filepaths)
        logging.info("FINISHED adding census features.")

        return

    if not os.path.exists(DIAGNOSES_TO_CODES_PROCESSED_PATH):
        raise FileNotFoundError(
            "Supply the final processed ICD-10/ICD-9 definition table."
        )
    if not os.path.exists(PROCESSED_DEP_INDEX_PATH):
        raise FileNotFoundError("Supply the final population-weighted ZIP-3 ADI table.")
    if patient_tokens_file is None:
        raise ValueError("Supply the verified cohort membership CSV with -ptf.")

    # Step 2: create batch folders
    skip_batch_range_creation = True
    for i in range(num_batches):
        if not os.path.exists(
            os.path.join(data_batch_path, str(i), BATCH_RANGE_FILENAME)
        ):
            skip_batch_range_creation = False

    if skip_batch_range_creation:
        logging.info(
            "Batch folders with batch range files already exist! SKIPPING this creation step..."
        )
    elif multiple_servers:
        raise Exception(
            "Batch folder creation must be complete before running this on multiple servers."
        )
    else:
        min_patient_id, max_patient_id = get_patient_id_range(
            os.path.join(data_path, cohort_folder_name), cohort_filenames
        )

        write_batch_folders(
            num_batches, min_patient_id, max_patient_id, data_batch_path
        )

    # Step 3: For each batch, write a list of raw data files that contain its data
    skip_batch_to_raw_file_creation = True
    for i in range(num_batches):
        if not os.path.exists(
            os.path.join(data_batch_path, str(i), BATCH_TO_FILES_FILENAME)
        ):
            skip_batch_to_raw_file_creation = False

    if skip_batch_to_raw_file_creation:
        logging.info(
            "Batch folders with batch to raw filename files already exist! SKIPPING this creation step..."
        )
    elif multiple_servers:
        raise Exception(
            "batch to raw filename creation must be complete before running this on multiple servers."
        )
    else:
        handle_batch_to_raw_files_mappings(
            data_batch_path,
            medical_header_filepaths
            + static_filepaths
            + cohort_filepaths
            + filtered_visit_filepaths
            + closed_filepaths,
            num_processes=num_batches,
        )

    if prepare_only:
        logging.info(
            "Preparation complete. Batch files are ready for medical-header processing."
        )
        return

    # Step 4: Process each medical header file
    process_medical_header_files(
        data_batch_path, num_batches=num_batches, num_processes=num_processes
    )
    if multiple_servers:
        logging.info(
            "FINISHED processing medical header files. Change script options to single server to continue."
        )
        return

    create_batch_closed_data_dics(
        data_batch_path,
        num_batches=num_batches,
        num_processes=num_processes,
    )

    # Step 5: Process visits files
    create_batch_visits_data_dics(
        data_batch_path,
        num_batches=num_batches,
        num_processes=num_processes,
    )

    # Step 7: Process static files
    create_batch_static_data_dics(
        data_batch_path,
        num_batches=num_batches,
        num_processes=num_processes,
    )

    # Step 8: Combine all outcomes in a single batch. Also handle death in this function.
    create_batch_data_dics(
        data_batch_path, num_batches=num_batches, num_processes=num_processes
    )

    # Step 9: Combine all batches into a single data dic
    combine_all_batch_data_dics(data_batch_path, num_batches=num_batches)

    # Step 10: Final processing. Filter to subset of patients, subset of treatment groups,
    # Apply indicator function, split data_dic into X and Y.
    processed_data_dic_path = apply_final_processing(
        final_output_location,
        data_batch_path,
        patient_tokens_file=patient_tokens_file,
    )

    # Step 11: add in census features: income, education, and ethnicity

    add_census_features(processed_data_dic_path, static_filepaths)

    logging.info("FINISHED preprocessing pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Preprocess_Pipeline",
        description="""Pipeline to preprocess the data from its raw format to its sparse format""",
    )
    parser.add_argument(
        "-nb",
        "--num_batches",
        type=int,
        default=256,
        help="The number of batches to split the data into.",
    )
    parser.add_argument(
        "-np",
        "--num_processes",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Number of worker processes for preprocessing.",
    )
    parser.add_argument(
        "-sl",
        "--suppress_logging",
        action="store_true",
        help="Set this to suppress logging statements.",
    )
    parser.add_argument(
        "-ms",
        "--multiple_servers",
        action="store_true",
        help="Run the medical-header stage only, after --prepare-only has completed.",
    )

    parser.add_argument(
        "-ptf",
        "--patient_tokens_file",
        type=str,
        required=True,
        help="Verified cohort membership CSV with PATIENT_ID and binary TREATMENT.",
    )

    parser.add_argument(
        "-fl",
        "--final_output_location",
        type=str,
        default="final",
        help="Output location for final data dict.",
    )

    parser.add_argument(
        "-co",
        "--cohort_folder_name",
        type=str,
        default=COHORT_FOLDERNAME,
        help="Optional filepath for where the cohort is located.",
    )

    parser.add_argument(
        "-bf",
        "--raw_data_batch_folder",
        type=str,
        default=RAW_DATA_BATCH_FOLDER,
        help="Optional filepath for where the batch folder.",
    )

    parser.add_argument(
        "-o",
        "--only_census_step",
        action="store_true",
        default=False,
        help="Add Census linkage to an existing preprocessed cohort.",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after preparing diagnosis and batch mappings (steps 1-3).",
    )
    args = parser.parse_args()
    if args.prepare_only and (args.multiple_servers or args.only_census_step):
        parser.error("--prepare-only cannot be combined with -ms or -o")
    logging.info(f"args: {args}")

    if args.suppress_logging:
        # This is set to ERROR, otherwise the amount of logging will slow the task significantly
        logging.getLogger().setLevel(logging.ERROR)

        # Ignore warnings from Pandas
        pd.options.mode.chained_assignment = None  # default='warn'

    main(
        args.num_batches,
        args.num_processes,
        args.multiple_servers,
        args.patient_tokens_file,
        args.final_output_location,
        args.cohort_folder_name,
        args.raw_data_batch_folder,
        args.only_census_step,
        args.prepare_only,
    )

import os
import time
# import shutil
import concurrent.futures

# Importing necessary functions and classes
from src.utils.db_utils import run_query
from src.utils.patient import Patient
import pandas as pd
import math

# Function to retrieve and process cohort data for given hospital admission IDs (hadm_ids)
def get_info_cohort(hadm_ids, save_path, parallel=True, globaldf_exists=False):
    time_start = time.time()
    # SQL query to retrieve chart events and relevant items for the specified hadm_ids
    query = """
    SELECT CE.subject_id, CE.hadm_id, CE.charttime, CE.value, D.label
    FROM CHARTEVENTS CE
    JOIN D_ITEMS D 
        ON CE.itemid = D.itemid
    WHERE hadm_id IN %(hadm_ids)s
    """

    # Run the query to get patient data based on hadm_ids
    df_patients = run_query(query, {"hadm_ids": tuple(hadm_ids)})

    print(f"SQL request done in {time.time() - time_start} seconds")

    # Process each patient record either in parallel or sequentially
    if parallel:
        # Using ThreadPoolExecutor for parallel processing of patients
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(Patient, df_patient, save_path)
                for hadm_id, df_patient in df_patients.groupby("hadm_id")
            ]
            concurrent.futures.wait(futures)
    else:
        # Sequential processing if parallel is set to False
        for hadm_id, df_patient in df_patients.groupby("hadm_id"):
            print(hadm_id)
            Patient(df_patient, save_path)

# Function to retrieve a cohort based on ICD-9 code, process them in batches, and save to specified path
def get_df_cohort(icd9_code, max_lim=10, batch_size=10, save_path="data/MIMIC/cohorts_new/", parallel=True):
    # SQL query to retrieve distinct hospital admission IDs for patients with a specified ICD-9 code
    query = """
    SELECT DISTINCT icustays.hadm_id
    FROM icustays
    JOIN DIAGNOSES_ICD ON DIAGNOSES_ICD.hadm_id = icustays.hadm_id
    WHERE icd9_code = '%(ARDS_list)s'
    """

    # Run the query to get hadm_ids of patients matching the ICD-9 code
    hadm_ids = run_query(query, {"ARDS_list": icd9_code})["hadm_id"].tolist()

    print(f"Getting cohort for code {icd9_code}, {len(hadm_ids)} patients have been found")

    if max_lim is None:
        max_lim = len(hadm_ids)

    # Function to process a batch of patients and save their data
    def process_batch(i):
        if i * batch_size >= max_lim:
            return
        get_info_cohort(hadm_ids[i * batch_size:(i + 1) * batch_size], save_path)

    # Display total batches and start data download, either in parallel or sequentially
    print(f"Starting data download : {max_lim} patients, so {max_lim // batch_size} batches of {batch_size} patients.")
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_batch, range(max_lim // batch_size))
    else:
        for i in range(max_lim // batch_size):
            t = time.time()
            process_batch(i)
            print(f"Batch {i} finished in {time.time() - t} seconds.")

# Main function to handle cohort data file management and cleanup
def main(project_dir):
    # Load patient data from a CSV file
    df_patients = pd.read_csv(os.path.join(project_dir, "patients.csv"))
    print(len(df_patients))
    df_patients = df_patients.drop_duplicates(subset=["subject_id", "hadm_id"])  # Remove duplicates
    print(f"{len(df_patients)} patients in cohort")

    # Check if patient data already exists in the save directory; if so, remove from dataframe
    for index, row in df_patients.iterrows():
        try:
            if os.path.exists(os.path.join(project_dir, str(int(row["subject_id"])), str(int(row["hadm_id"])))):
                df_patients.drop(index=index, inplace=True)
        except ValueError:
            print("Value error", index)

    print(f"{len(df_patients)} patients left to add")

# Entry point for running the script
if __name__ == '__main__':
    save_path = "data/MIMIC/final"  # Define save path for cohort data
    main(save_path)  # Call main function with the save path

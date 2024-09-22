import pandas as pd
from sqlalchemy import create_engine
import os
import time
import shutil
import json
import concurrent.futures
from src.utils.db_utils import run_query
from src.utils.patient import Patient


def get_info_cohort(hadm_ids, save_path, parallel=True):
    time_start = time.time()
    query = """
    SELECT CE.subject_id, CE.hadm_id, CE.charttime, CE.value, D.label
    FROM CHARTEVENTS_COHORTS CE
    JOIN D_ITEMS D 
        ON CE.itemid = D.itemid
    WHERE hadm_id IN %(hadm_ids)s
    """

    df_patients = run_query(query, {"hadm_ids": tuple(hadm_ids)})

    print(f"SQL request done in {time.time() - time_start} seconds")

    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(Patient, df_patient, save_path)
                for hadm_id, df_patient in df_patients.groupby("hadm_id")
            ]
            concurrent.futures.wait(futures)
    else:
        for hadm_id, df_patient in df_patients.groupby("hadm_id"):
            Patient(df_patient, save_path)


def get_df_cohort(icd9_code, max_lim=10, batch_size=10, save_path = "data/MIMIC/cohorts_new/", parallel=True):
    query = """
    SELECT DISTINCT icustays.hadm_id
    FROM icustays
    JOIN DIAGNOSES_ICD ON DIAGNOSES_ICD.hadm_id = icustays.hadm_id
    WHERE icd9_code = '%(ARDS_list)s'
    """

    hadm_ids = run_query(query, {"ARDS_list": icd9_code})["hadm_id"].tolist()

    print(f"Getting cohort for code {icd9_code}, {len(hadm_ids)} patients have been found")

    if max_lim is None:
        max_lim = len(hadm_ids)
    def process_batch(i):
        if i * batch_size >= max_lim:
            return
        get_info_cohort(hadm_ids[i * batch_size:(i + 1) * batch_size], save_path)

    print(f"Starting data download : {max_lim} patients, so {max_lim // batch_size} batches of {batch_size} patients.")
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_batch, range(max_lim // batch_size))
    else:
        for i in range(max_lim // batch_size):
            t = time.time()
            process_batch(i)
            print(f"Batch {i} finished in {time.time() - t} seconds.")


if __name__ == '__main__':
    save_path = "data/MIMIC/main/"
    ARDS_list = (51881, 51882, 51884, 5185, 51851)

    # ARDS_list = [51882]

    # TODO
    # check P/F
    # check PEEP
    # Check Articles
    # Check diagnostics clustering
    # Run estimators


    # Matt : 5185 12882
    # Other studies : other non ARDS codes -> they just predict the Pf ratio PEEP and radiology reports
    # Look at the sampling frequency of PEEP and P/F ratios when they are available

    delete_existing = False
    if delete_existing:
        shutil.rmtree(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for icd9_code in ARDS_list:
        df_cohort = get_df_cohort(icd9_code, max_lim=None, batch_size=400, save_path=save_path)

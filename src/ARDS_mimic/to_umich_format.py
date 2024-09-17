import pandas as pd
from sqlalchemy import create_engine
import os
import time
import shutil
import json
import concurrent.futures
from src.utils.db_utils import run_query
from src.utils.patient import Patient


def get_info_cohort(hadm_ids, save_path):
    time_start = time.time()

    query = """
    SELECT *
    FROM CHARTEVENTS
    JOIN D_ITEMS ON CHARTEVENTS.itemid = D_ITEMS.itemid
    WHERE hadm_id IN %(hadm_ids)s
    """

    df_patients = run_query(query, {"hadm_ids": tuple(hadm_ids)})

    print(f"SQL request done in {time.time() - time_start} seconds")
    for i, hadm_id in enumerate(hadm_ids):
        df_patient = df_patients[df_patients["hadm_id"] == hadm_id]
        Patient(df_patient, save_path)


def get_df_cohort(icd9_codes, max_lim=10, batch_size=10, save_path = "data/MIMIC/cohorts_new/"):
    query = """
    SELECT DISTINCT icustays.hadm_id
    FROM icustays
    JOIN DIAGNOSES_ICD ON DIAGNOSES_ICD.hadm_id = icustays.hadm_id
    WHERE icd9_code IN %(ARDS_list)s
    """

    hadm_ids = run_query(query, {"ARDS_list": icd9_codes})["hadm_id"].tolist()

    if max_lim is None:
        max_lim = len(hadm_ids)

    def process_batch(i):
        if i * batch_size >= max_lim:
            return
        get_info_cohort(hadm_ids[i * batch_size:(i + 1) * batch_size], save_path)
        print(f"Batch {i} finished")

    print(f"Starting data download : {max_lim} patients, so {max_lim // batch_size} batches of {batch_size} patients.")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_batch, range(max_lim // batch_size))


if __name__ == '__main__':
    save_path = "data/MIMIC/cohorts_new/"
    # ARDS_list = ('51881', '51882', '51884', '51851', '51852', '51853', '769')
    ARDS_list = ('51882', 'None')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    # get_info_cohort([100108], save_path)
    df_cohort = get_df_cohort(ARDS_list, max_lim=100, batch_size=10, save_path=save_path)

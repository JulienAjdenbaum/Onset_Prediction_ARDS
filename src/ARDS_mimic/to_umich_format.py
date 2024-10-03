import os
import time
# import shutil
import concurrent.futures
#
# from src.utils.create_cohort_table import batch_size
from src.utils.db_utils import run_query
from src.utils.patient import Patient
import pandas as pd
import math

def get_info_cohort(hadm_ids, save_path, parallel=True, globaldf_exists=False):
    time_start = time.time()
    query = """
    SELECT CE.subject_id, CE.hadm_id, CE.charttime, CE.value, D.label
    FROM CHARTEVENTS CE
    JOIN D_ITEMS D 
        ON CE.itemid = D.itemid
    WHERE hadm_id IN %(hadm_ids)s
    """

    # print(tuple(hadm_ids))
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
            print(hadm_id)
            Patient(df_patient, save_path)


def get_df_cohort(icd9_code, max_lim=10, batch_size=10, save_path="data/MIMIC/cohorts_new/", parallel=True):
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
    save_path = "data/MIMIC/final"
    # ARDS_list = (51881, 51882, 51884, 5185, 51851)

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

    # delete_existing = False
    # if delete_existing:
    #     shutil.rmtree(save_path)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # for icd9_code in ARDS_list:
    #     df_cohort = get_df_cohort(icd9_code, max_lim=None, batch_size=400, save_path=save_path)

    df_patients = pd.read_csv(os.path.join(save_path, "patients.csv"))
    print(len(df_patients))
    df_patients = df_patients.drop_duplicates(subset=["subject_id", "hadm_id"])
    # print(len(df_patients))
    print(f"{len(df_patients)} patients in cohort")
    #
    for index, row in df_patients.iterrows():
        try:
            if os.path.exists(os.path.join(save_path, str(int(row["subject_id"])), str(int(row["hadm_id"])))):
                # print(index, "exists")
                df_patients.drop(index=index, inplace=True)
            # else:
            #     print(index, "not exists, removing")
        except ValueError:
            print("Value error", index)

    print(f"{len(df_patients)} patients left to add")
    #
    # def process_batch(i):
    #     if i * batch_size >= max_lim:
    #         return
    #     get_info_cohort(df_patients["hadm_id"].iloc[i * batch_size:min((i + 1) * batch_size, max_lim)], save_path,
    #                     globaldf_exists=True)
    #
    # batch_size = 400
    # max_lim = None
    # if max_lim is None:
    #     max_lim = len(df_patients)
    #
    #
    # print(f"Starting data download : {max_lim} patients, batch size {batch_size}, so {math.ceil(max_lim / batch_size)} batches")
    #
    # for i in range(max_lim // batch_size + 1):
    #     t = time.time()
    #     process_batch(i)
    #     # print(
    #     #     f"Batch {i} finished in {time.time() - t} seconds, now there is a total of {len(os.listdir(save_path))-1} patients saved")
    #
    # # print(os.listdir(save_path))
    # for hadm_id, subject_id in zip(df_patients["hadm_id"][:50], df_patients["subject_id"][:50]):
    #     print(
    #         f"{subject_id}, {int(hadm_id)} : {os.path.exists(os.path.join(save_path, str(int(subject_id)), str(int(hadm_id))))}")

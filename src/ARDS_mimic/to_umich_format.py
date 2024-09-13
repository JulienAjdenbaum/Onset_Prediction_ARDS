import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import psycopg2
from sqlalchemy import create_engine
import os
import time
import shutil
import json
import concurrent.futures
# from src.GMM.GMM_MIMIC import patient_id

# information used to create a database connection
sqluser = 'postgres'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to postgres with a copy of the MIMIC-III database
# con = psycopg2.connect(dbname=dbname, user=sqluser, password=sqluser)
engine = create_engine(f'postgresql://{sqluser}:{sqluser}@localhost/{dbname}')

# the below statement is prepended to queries to ensure they select from the right schema
query_schema = 'set search_path to ' + schema_name + ';'


def get_diagnoses(patient_id, hadm_id):
    query = query_schema + """
        SELECT long_title
        FROM DIAGNOSES_ICD
        JOIN D_ICD_DIAGNOSES ON DIAGNOSES_ICD.icd9_code = D_ICD_DIAGNOSES.icd9_code
        WHERE subject_id = {subject_id} AND hadm_id = {hadm_id}
        """

    df = pd.read_sql_query(query.format(subject_id=patient_id, hadm_id=hadm_id), engine)
    return list(df["long_title"])


def get_info_cohort(hadm_ids, query, df, batch):
    save_path = "data/MIMIC/cohorts_new/"
    hadm_ids_str = ','.join([str(h) for h in hadm_ids])
    time_start = time.time()
    df_patient_mimic_format_all = pd.read_sql_query(query.format(hadm_ids=hadm_ids_str), engine)
    print(f"SQL request done in {time.time() - time_start} seconds")

    for i, hadm_id in enumerate(hadm_ids):
        subject_id = df[df["hadm_id"] == hadm_id]["subject_id"].values[0]
        print(f"Getting data for subject {subject_id} from batch {batch}")

        patient_path = os.path.join(save_path, f"{subject_id}")
        if not os.path.exists(patient_path):
            os.mkdir(patient_path)

        # print(patient_path, subject_id)
        encounter_id = os.listdir(patient_path)
        if len(encounter_id) > 0:
            encounter_id = int(encounter_id[-1]) + 1
        else:
            encounter_id = 0

        encounter_path = os.path.join(patient_path, f"{encounter_id}")
        if not os.path.exists(encounter_path):
            os.mkdir(encounter_path)

        diagnoses = get_diagnoses(subject_id, hadm_id)

        df_patient_mimic_format = df_patient_mimic_format_all[df_patient_mimic_format_all["hadm_id"] == hadm_id]

        time_start = df_patient_mimic_format["charttime"].min()
        rows = []

        for index, row in df_patient_mimic_format.iterrows():
            # print(row)
            time_feature = (row["charttime"] - time_start).total_seconds() / 3600
            label = row["label"]
            value = row["value"]
            rows.append(({"patient_id": subject_id, "encounter_id": encounter_id, "hadm_id": hadm_id, "time": time_feature, label: value}))

        df_patient = pd.DataFrame(rows).sort_values(by="time").reset_index(drop=True)
        df_patient = df_patient.ffill()

        df_patient.to_csv(encounter_path + "/patient_mimic_format.csv", index=False)

        config = {}
        config["diagnoses"] = diagnoses
        # config["hadm_id"] = hadm_id

        with open(encounter_path + "/diagnoses.json", "w") as f:
            json.dump(config, f, indent=4)



def get_df_cohort(icd9_codes, path="", max_lim = 10, batch_size = 10):
    query = query_schema + """
    SELECT DISTINCT icustays.subject_id, icustays.hadm_id
    FROM icustays
    JOIN DIAGNOSES_ICD ON DIAGNOSES_ICD.hadm_id = icustays.hadm_id
    WHERE icd9_code IN {ARDS_list}
    """

    df = pd.read_sql_query(query.format(ARDS_list=icd9_codes), engine)

    query = query_schema + """
    SELECT *
    FROM CHARTEVENTS
    JOIN D_ITEMS ON CHARTEVENTS.itemid = D_ITEMS.itemid
    WHERE hadm_id IN ({hadm_ids})
    """

    if max_lim is None:
        max_lim = len(df)

    hadm_ids = df["hadm_id"].unique()

    def process_batch(i):
        if i * batch_size >= max_lim:
            return
        get_info_cohort(hadm_ids[i * batch_size:(i + 1) * batch_size], query, df, i)
    print(f"Starting data download : {max_lim} patients, so {max_lim // batch_size} batches of {batch_size} patients.")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # executor.map(process_batch, range(len(df) // batch_size))
        executor.map(process_batch, range(max_lim // batch_size))

    # for i in range(int(len(df)//batch_size)):
    #     process_batch(i)




if __name__ == '__main__':
    save_path = "data/MIMIC/cohorts_new/"
    # ARDS_list = ('51881', '51882', '51884', '51851', '51852', '51853', '769')
    ARDS_list = ('51882', 'None')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    df_cohort = get_df_cohort(ARDS_list, path=save_path, max_lim=10)
    # df_cohort.to_csv(f"data/MIMIC/cohorts/{ARDS_list}.csv", index=False)
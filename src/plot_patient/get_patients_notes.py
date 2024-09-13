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
import math

# from src.GMM.GMM_MIMIC import patient_id

# information used to create a database connection
sqluser = 'postgres'
dbname = 'mimic'
schema_name = 'mimiciii'

engine = create_engine(f'postgresql://{sqluser}:{sqluser}@localhost/{dbname}')

query_schema = 'set search_path to ' + schema_name + ';'
caregivers = {}


def get_patient_idm_date(hadm_id):
    query = query_schema + """
            SELECT admittime
            FROM ADMISSIONS
            WHERE hadm_id = {hadm_id}
            LIMIT 1
        """

    df = pd.read_sql_query(query.format(hadm_id=hadm_id), engine)
    return df["admittime"].iloc[0]


# def get_title(df):
#     i, df = df
#     return f"Note {i}: {df['charttime']}\t{df['category']}\t{df['cgid']}\t{df['description']}"


def get_patients_notes(hadm_id):
    query = query_schema + """
        SELECT charttime, text, category, cgid, description
        FROM NOTEEVENTS
        WHERE hadm_id = {hadm_id}
    """
    caregivers = {}
    def get_nice_caregiver_name(cgid):
        if math.isnan(cgid):
            return "Unknown caregiver"

        query = query_schema + """
            SELECT label
            FROM CAREGIVERS
            WHERE cgid = {cgid}
        """

        caregiver = pd.read_sql_query(query.format(cgid=cgid), engine).values[0, 0]
        if caregiver is None:
            return "Unknown caregiver"
        if caregiver not in caregivers:
            caregivers[caregiver] = [cgid]
        if cgid not in caregivers[caregiver]:
            caregivers[caregiver].append(cgid)
        caregiver = str(caregiver) + " " + str(caregivers[caregiver].index(cgid) + 1)
        return caregiver

    df = pd.read_sql_query(query.format(hadm_id=hadm_id), engine)
    admit_time = get_patient_idm_date(hadm_id)
    df["charttime"] = df["charttime"] - admit_time
    df["cgid"] = list(map(get_nice_caregiver_name, list(df["cgid"])))
    return df["category"], df["description"], df["charttime"], df["cgid"], df["text"]


if __name__ == '__main__':
    hadm_id = 164853
    # print(get_patient_idm_date(hadm_id))
    get_patients_notes(hadm_id)

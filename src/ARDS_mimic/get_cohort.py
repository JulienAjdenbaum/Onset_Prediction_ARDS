import pandas as pd
import os
from src.utils.patient import Patient
from src.utils.db_utils import run_query
import time
import matplotlib.pyplot as plt
import numpy as np
import shutil

def select_cohort(df, critical_measurements, min_age=18, min_los=20, max_los=1000):
    print(f"Number of admissions in MIMIC III: {54432}")
    print(f"Number of admissions after dataset origin criteria: {len(df)}")
    df = df[df["age (years)"] >= min_age]
    print(f"Number of admissions after age criteria: {len(df)}")
    df = df[(df["los (hrs)"] > min_los) & (df["los (hrs)"] <= max_los)]
    print(f"Number of admissions after los criteria: {len(df)}")
    print()
    for critical_measurement in critical_measurements:
        print(f"Number of admissions with at least one measurement for {critical_measurement}: {len(df)-df[critical_measurement].isna().sum()}")
        # print(df[critical_measurement].dropna().mean())
    df = df.dropna()
    print()
    print(f"Final number of admissions kept: {len(df)}")
    return df


def add_PEEP(patient: Patient):
    raw_df = patient.raw_df
    print(raw_df.columns)


def add_age(df):
    if "age (years)" in df.columns:
        print("Age already in df")
        return df
    hadm_ids = tuple(df['hadm_id'].astype(str).tolist())

    # SQL query to get age for all relevant hadm_ids
    query = f"""
        SELECT admissions.hadm_id, (admissions.admittime - patients.dob) AS age
        FROM patients
        JOIN admissions
            ON patients.subject_id = admissions.subject_id
        WHERE admissions.hadm_id IN {hadm_ids}
    """

    age_df = run_query(query, {"hadm_ids": hadm_ids}).drop_duplicates()
    for index, row in age_df.iterrows():
        age_df.loc[index, 'age (years)'] = row['age'].days / 365

    return df.merge(age_df[['hadm_id', 'age (years)']], on='hadm_id', how='left')


def add_los(df):
    if "los (hrs)" in df.columns:
        print("Los already in df")
        return df
    hadm_ids = tuple(df['hadm_id'].astype(str).tolist())

    # SQL query to get age for all relevant hadm_ids
    query = f"""
            SELECT hadm_id, (admissions.dischtime - admissions.admittime) AS los
            FROM admissions
            WHERE admissions.hadm_id IN {hadm_ids}
        """
    los_df = run_query(query, {"hadm_ids": hadm_ids}).drop_duplicates()

    los_df['los'] = pd.to_timedelta(los_df['los'], errors='coerce')
    los_df['los (hrs)'] = los_df['los'].dt.seconds / 3600 + los_df['los'].dt.days * 24

    return df.merge(los_df[['hadm_id', 'los (hrs)']], on='hadm_id', how='left')


def add_critical_measurement_count(df, critical_measurements, dataset):
    query = """
        SELECT itemid, label
        FROM d_items
        WHERE label in %(critical_measurements)s AND dbsource = %(dataset)s
    """
    itemid_df = run_query(query, {"critical_measurements": tuple(critical_measurements), "dataset": dataset}).drop_duplicates()

    hadm_ids = tuple(df['hadm_id'].astype(str).tolist())

    for index, row in itemid_df.iterrows():

        measurement_name = row['label']
        measurement_itemid = row['itemid']

        if measurement_name in df.columns:
            print(f"{measurement_name} already in df")
            continue

        query = f"""
                SELECT hadm_id, COUNT(*) AS measurement
                FROM chartevents
                WHERE itemid = %(measurement_itemid)s AND hadm_id IN %(hadm_ids)s
                GROUP BY hadm_id
            """

        measurement_df = run_query(query, {"hadm_ids": hadm_ids, "measurement_itemid": measurement_itemid}).drop_duplicates()
        measurement_df.rename(columns={'measurement': measurement_name}, inplace=True)

        df = df.merge(measurement_df[['hadm_id', measurement_name]], on='hadm_id', how='left')
        # print(df)

    return df
def add_dataset_origin(df):
    if "dataset" in df.columns:
        print("Dataset origin already in df")
        return df
    hadm_ids = tuple(df['hadm_id'].astype(str).tolist())

    # SQL query to get age for all relevant hadm_ids
    query = f"""
            SELECT icustays.hadm_id, dbsource as dataset
            FROM icustays
            WHERE icustays.hadm_id IN {hadm_ids}
        """

    db_df = run_query(query, {"hadm_ids": hadm_ids}).drop_duplicates()
    return df.merge(db_df[['hadm_id', 'dataset']], on='hadm_id', how='left')


def create_patient_df():
    query = """
    SELECT subject_id, hadm_id
    FROM admissions
    """
    df = run_query(query, {}).drop_duplicates()
    return df


if __name__ == "__main__":
    plot = False
    save = True
    dataset = "carevue"
    project_dir = "data/MIMIC/final"

    critical_measurements = ["Heart Rate", "SpO2", "Respiratory Rate", "Arterial BP [Systolic]", "Arterial BP [Diastolic]",
                             "Temperature F", 'PEEP Set', 'FiO2 Set', 'Arterial PaO2']

    # if os.path.exists(os.path.join(project_dir, "patients.csv")):
    #     os.remove(os.path.join(project_dir, "patients.csv"))
    #     patients_list_df = create_patient_df()
    # else:
    #     if not os.path.exists(os.path.join(project_dir)):
    #         os.mkdir(project_dir)
    #     patients_list_df = create_patient_df()

    # if os.path.exists(project_dir):
    #     shutil.rmtree(project_dir)

    os.mkdir(project_dir)

    patients_list_df = create_patient_df()
    print(len(patients_list_df))
    patients_list_df = add_age(patients_list_df)
    print(len(patients_list_df))
    patients_list_df = add_los(patients_list_df)
    print(len(patients_list_df))
    patients_list_df = add_dataset_origin(patients_list_df)
    print(len(patients_list_df))
    patients_list_df = add_critical_measurement_count(patients_list_df, critical_measurements, dataset)
    if plot or True:
        print(patients_list_df)
    print()
    patients_list_df = select_cohort(patients_list_df, critical_measurements)

    if plot:
        plt.hist(patients_list_df['los (hrs)'], bins=100)
        plt.show()
        plt.hist(patients_list_df['age (years)'], bins=100)
        plt.show()

    # duplicates = patients_list_df[patients_list_df.duplicated(subset=['hadm_id', 'subject_id'], keep=False)]
    #
    #
    # # Get the number of duplicates
    # num_duplicates = duplicates.shape[0]
    # print(patients_list_df.duplicated(subset=['hadm_id', 'subject_id'], keep=False).sum())

    if save:
        patients_list_df.to_csv(os.path.join(project_dir, "patients.csv"))

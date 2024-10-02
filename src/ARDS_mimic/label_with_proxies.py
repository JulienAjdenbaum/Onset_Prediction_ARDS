import shutil
import warnings
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import re
import json
from matplotlib.animation import FuncAnimation
from src.LLM_labeling.get_LLM_label import get_LLM_result
from src.utils.patient import Patient
import concurrent.futures
from functools import partial
import time
import cProfile
import pstats


def get_label_by_proxies(patient):
    try:
        df = patient.get_processed_df()
    except AttributeError:
        return -1
    # print(df.columns)
    df["P/F ratio"] = df["Arterial PaO2"] / df["FiO2 Set"]
    labels = np.argwhere((df["P/F ratio"] < 300 ) & (df["PEEP Set"] > 5))
    # print(labels)
    if len(labels) == 0:
        return -1
    return labels[0][0]


if __name__ == "__main__":
    project_dir = "/home/julien/Documents/stage/data/MIMIC/full"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))
    columns_to_keep = ["Heart Rate", "SpO2", "Respiratory Rate", "Arterial BP [Systolic]", "Arterial BP [Diastolic]",
                             "Temperature F", 'PEEP Set', 'FiO2 Set', 'Arterial PaO2']
    t = time.time()
    labels = []
    patient_ARDS_df = pd.DataFrame()
    for index, row in patients_list_df.iterrows():
        patient = Patient.load(project_dir, str(int(row["subject_id"])), str(int(row["hadm_id"])))
        # try:
        print(row["subject_id"], row["hadm_id"])
        label = get_label_by_proxies(patient)
        labels.append(label)
        if label != -1:
            config = patient.get_existing_config()
            config["proxy_label"] = str(label)
            patient.save_config(config)
            print("config_saved")
            patient_ARDS_df = pd.concat(
                [patient_ARDS_df, pd.DataFrame({"subject_id": [patient.subject_id], "hadm_id": [patient.hadm_id]})])

    patient_ARDS_df.to_csv(os.path.join(project_dir, "patient_ARDS_df.csv"), index=False)
            # else:
            #     patient.del_patient()
        # except Exception as e:
        #     print(e)
            # try:
            #     patient.del_patient()
            # except:
            #     pass


    labels = np.array(labels)
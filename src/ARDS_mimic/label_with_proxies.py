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


# Function to assign a label to a patient based on proxy criteria for ARDS diagnosis
def get_label_by_proxies(patient):
    try:
        # Load the processed data frame for the patient
        df = patient.get_processed_df()
    except AttributeError:
        return -1

    # Calculate P/F ratio (PaO2/FiO2 ratio) for ARDS proxy
    df["P/F ratio"] = df["Arterial PaO2"] / df["FiO2 Set"]

    # Identify rows where P/F ratio is below 300 and PEEP is above 5 (criteria for ARDS)
    labels = np.argwhere((df["P/F ratio"] < 300) & (df["PEEP Set"] > 5))

    # Return -1 if no rows meet the criteria, indicating no ARDS label
    if len(labels) == 0:
        return -1

    # Use the earliest time point that meets the criteria as the ARDS onset label
    label = df["time"].iloc[labels[0][0]]
    print(df["time"].min(), df["time"].max(), label)
    return label


if __name__ == "__main__":
    # Define project directory path and load list of patients from CSV file
    project_dir = "/home/julien/Documents/stage/data/MIMIC/final"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))

    # Define columns to keep for processing
    columns_to_keep = ["Heart Rate", "SpO2", "Respiratory Rate", "Arterial BP [Systolic]", "Arterial BP [Diastolic]",
                       "Temperature F", 'PEEP Set', 'FiO2 Set', 'Arterial PaO2']

    t = time.time()  # Start timer for processing
    labels = []  # List to store labels for each patient
    patient_ARDS_df = pd.DataFrame()  # DataFrame to store patient IDs with ARDS labels

    # Iterate through each patient in the list
    for index, row in patients_list_df.iterrows():
        # Load patient data using their subject and hospital admission IDs
        patient = Patient.load(project_dir, str(int(row["subject_id"])), str(int(row["hadm_id"])))

        # Get ARDS onset time label based on proxy criteria
        label = get_label_by_proxies(patient)
        labels.append(label)  # Append label (or -1 if no label) to list

        # If a valid label was found, save it to the patient’s configuration and add to ARDS patient DataFrame
        if label != -1:
            config = patient.get_existing_config()
            config["proxy_label"] = str(label)  # Save label as proxy label in config
            patient.save_config(config)  # Save updated config to the patient's data

            # Add patient to ARDS DataFrame for final output
            patient_ARDS_df = pd.concat(
                [patient_ARDS_df, pd.DataFrame({"subject_id": [patient.subject_id], "hadm_id": [patient.hadm_id]})])

    # Save DataFrame of patients with ARDS labels to a CSV file
    patient_ARDS_df.to_csv(os.path.join(project_dir, "patient_ARDS_df.csv"), index=False)

    # Convert labels list to numpy array
    labels = np.array(labels)

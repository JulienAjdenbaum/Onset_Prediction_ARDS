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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

keys_number_dic = {}


def get_keys(patient: Patient, keys_list):
    config = patient.get_existing_config()
    if "ards_onset_time" not in config:
        return False
    ards_onset_time = config["ards_onset_time"]
    if ards_onset_time is None or ards_onset_time < 1:
        return False
    if "fit_keys" not in config.keys():
        return False
    keys = config["fit_keys"]
    for key in keys_list:
        if key not in keys:
            return False
    return True

def to_diff(df, mode="diff"):
    if mode == "diff":
        df_diff = df.diff()[1:]
        return df_diff.drop("time", axis=1).div(df_diff.time, axis=0)
    if mode == "concat":
        df_diff = df.diff()[1:]
        # print(df_diff)
        return pd.concat([df.drop("time", axis=1)[1:], df_diff.drop("time", axis=1).div(df_diff.time, axis=0)],axis=1)
    return df.drop("time", axis=1)



def get_samples(subjects, patient_list_df, time_before_ARDS, time_after_ARDS):
    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    samples_before_ARDS = None
    samples_after_ARDS = None

    for subject_id in subjects:
        hadm_id = patient_list_df[patient_list_df["subject_id"] == subject_id]["hadm_id"].values[0]
        patient = Patient.load(project_dir, str(subject_id), str(hadm_id))
        config = patient.get_existing_config()
        ards_onset_time = config["ards_onset_time"]
        df_final = patient.get_final_df()
        df_before = df_final[df_final["time"]<ards_onset_time-time_before_ARDS]
        df_after = df_final[(df_final["time"]>ards_onset_time) & (df_final["time"]<ards_onset_time+time_after_ARDS) ]
        print(f"{len(df_before)} samples before ARDS and {len(df_after)} samples after ARDS")

        if samples_before_ARDS is None:
            samples_before_ARDS = to_diff(df_before)
            # print(samples_before_ARDS.shape)
            # print(samples_before_ARDS)
        else:
            samples_before_ARDS = np.concatenate((samples_before_ARDS, to_diff(df_before)), axis=0)

        if samples_after_ARDS is None:
            samples_after_ARDS = to_diff(df_after)
        else:
            samples_after_ARDS = np.concatenate((samples_after_ARDS, to_diff(df_after)), axis=0)
    return samples_before_ARDS, samples_after_ARDS

def get_cohort(patients_list_df, keys_to_keep):
    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    patient_list = []
    for index, row in patients_list_df.iterrows():
        patient = Patient.load(project_dir, str(row["subject_id"]), str(row["hadm_id"]))
        config = patient.get_existing_config()
        if get_keys(patient, keys_to_keep) and len(config["fit_keys"]) == len(keys_to_keep):
            patient_list.append(patient.subject_id)
    df = pd.DataFrame({"subject_id": patient_list})
    df.to_csv("cohort.csv", index=False)


def run_logistic_regression(samples_before, samples_after):
    X = np.vstack((samples_before, samples_after))
    y = np.hstack((np.zeros(samples_before.shape[0]), np.ones(samples_after.shape[0])))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)
    return model




def test_logisitc_regression(subjects, patient_list_df, model, time_before_ARDS, time_after_ARDS):
    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    samples_before_ARDS = None
    samples_after_ARDS = None
    for subject_id in subjects:
        hadm_id = patient_list_df[patient_list_df["subject_id"] == subject_id]["hadm_id"].values[0]
        patient = Patient.load(project_dir, str(subject_id), str(hadm_id))
        config = patient.get_existing_config()
        ards_onset_time = config["ards_onset_time"]
        df_final = patient.get_final_df()
        df_diff = to_diff(df_final)
        scores = model.predict_proba(df_diff)[:, 0]
        df_before = df_final[df_final["time"]<ards_onset_time]
        df_after = df_final[df_final["time"]>ards_onset_time]
        print(f"{len(df_before)} samples before ARDS and {len(df_after)} samples after ARDS")

        if samples_before_ARDS is None:
            samples_before_ARDS = to_diff(df_before)
        else:
            samples_before_ARDS = np.concatenate((samples_before_ARDS, to_diff(df_before)), axis=0)

        if samples_after_ARDS is None:
            samples_after_ARDS = to_diff(df_after)
        else:
            samples_after_ARDS = np.concatenate((samples_after_ARDS, to_diff(df_after)), axis=0)

        plt.plot(df_final["time"][:len(scores)], scores, label="score")
        plt.axvline(ards_onset_time, linestyle="--", color="orange", label="ards_onset_time")
        plt.axvspan(0, ards_onset_time-time_before_ARDS, alpha=0.3, color="green", label="Negative time")
        plt.axvspan(ards_onset_time, ards_onset_time+ time_before_ARDS , alpha=0.3, color="red", label="ARDS time")
        plt.legend()
        plt.show()
        # break
    X_test = np.vstack((samples_before_ARDS, samples_after_ARDS))
    y_test = np.hstack((np.zeros(samples_before_ARDS.shape[0]), np.ones(samples_after_ARDS.shape[0])))

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Testing Accuracy: {test_accuracy:.2f}')



if __name__ == "__main__":
    time_before_ARDS = 4
    time_after_ARDS = 12

    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))
    keys_to_keep = ["Heart Rate", "SpO2", "Respiratory Rate", "NBP Mean", "NBP [Systolic]", "NBP [Diastolic]",
                    "Temperature F"]
    # get_cohort(patients_list_df, keys_to_keep)
    cohort = pd.read_csv("cohort.csv")['subject_id'].values
    train_subjects, test_subjects = train_test_split(cohort, test_size=0.2, random_state=0)

    # train_subjects = [train_subjects[0]]
    # test_subjects = train_subjects
    # print(len(train_subjects, test_subjects)
    print(f"Cohort loaded with {len(train_subjects)} training subjects and {len(test_subjects)} test subjects")
    samples_before_ards, samples_after_ards = get_samples(train_subjects, patients_list_df, time_before_ARDS, time_after_ARDS)
    np.save("samples_before.npy", samples_before_ards)
    np.save("samples_after.npy", samples_after_ards)

    samples_before_ards = np.load("samples_before.npy")
    samples_after_ards = np.load("samples_after.npy")
    print(f"{len(samples_before_ards)} samples before ARDS and {len(samples_after_ards)} samples after ARDS")
    model = run_logistic_regression(samples_before_ards, samples_after_ards)
    test_logisitc_regression(test_subjects, patients_list_df, model, time_before_ARDS, time_after_ARDS)

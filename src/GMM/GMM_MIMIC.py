import shutil
import warnings
from distutils.command.config import config
from statistics import covariance

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from pooch import retrieve
from skimage.feature import shape_index
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import re
import json
from matplotlib.animation import FuncAnimation
import imageio
from src.LLM_labeling.get_LLM_label import get_LLM_result

# def nan_percentage(group):
#     return group.isna().mean() * 100

def get_existing_config(project_dir, patient_id, encounter_id):
    path_save = os.path.join(project_dir, patient_id, encounter_id, "saves")
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    if os.path.exists(os.path.join(path_save, f"config.json")):
        with open(os.path.join(path_save, f"config.json"), "r") as f:
            config = json.load(f)
            if "analysis_parameters" not in config:
                config["analysis_parameters"] = {}
            return config
    if os.path.exists(os.path.join(project_dir, patient_id, encounter_id, "diagnoses.json")):
        with open(os.path.join(project_dir, patient_id, encounter_id, "diagnoses.json"), 'r') as f:
            config = json.load(f)
            if "analysis_parameters" not in config:
                config["analysis_parameters"] = {}
            return config
    warnings.warn(f"Patient {patient_id} encounter {encounter_id} does not have diagnoses file")
    return {"diagnoses": [], "analysis_parameters":{}}

def nan_percentage(column):
    return column.isna().mean() * 100

# Function to count changes for each column
def count_changes(df):
    return (df != df.shift()).sum()

def my_score(X, gm):
    scores = np.zeros(X.shape)
    std = np.sqrt(gm.covariances_)
    # print(std[0])
    # print(X.shape)
    for sample in range(X.shape[0]):
        for j in range(X.shape[1]):
            scores[sample, j] = (X[sample, j]/np.mean(std[:, j]))**2
    # print(np.mean(scores, axis=0))
    return np.mean(scores)

def get_ARDS_onset_time(hadm_id, save_path, start_time):
    patient_dir = save_path
    ARDS_onset_time_txt = get_LLM_result(patient_dir, hadm_id, start_time)["onset_timestamp"]
    # print(ARDS_onset_time_txt)
    if ARDS_onset_time_txt[0] == "-":
        return None
    else:
        ARDS_onset_time = int(ARDS_onset_time_txt.split(" ")[0]) * 24 + int(ARDS_onset_time_txt.split(" ")[-1])
        return ARDS_onset_time

def generate_patient_plot(df, patient_id, ARDS_onset_time, show, save_path, left_bar=None, right_bar=None):

    columns_interesting_left = ["Temperature F", "Heart Rate", "Respiratory Rate", "SpO2"]
    columns_interesting_right = ["NBP [Systolic]"]

    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot on the left y-axis
    ax1.set_xlabel('Time (hours)')
    # ax1.set_ylabel('col1 (left axis)', color='tab:blue')
    for col in columns_interesting_left:
        if col in df.columns:
            ax1.plot(df["time"], df[col], label=col)

    for col in columns_interesting_right:
        if col in df.columns:
            ax1.plot(df["time"], df[col], label=col)
    ax1.tick_params(axis='y')

    ax1.axvline(left_bar, label="t_fit_start", color="green", linestyle="--")
    ax1.axvline(right_bar, label="t_fit_end", color="red", linestyle="--")
    if ARDS_onset_time is not None:
        print(left_bar)
        ax1.axvline(ARDS_onset_time, label="ARDS_onset_time", color="orange", linestyle="--")

    ax1.legend(loc='upper left')

    # fig.tight_layout()
    plt.title(f"Data for patient {patient_id}")
    # print(save_path)
    plt.savefig(save_path)
    if show: plt.show()
    plt.close()
    # return ax1


def run_patient_encounter_preprocessing(patient_id, encounter_id, project_dir,
                                        file_name = "patient_mimic_format.csv", run_PCA = False, plot=False):

    limit_n_columns = True
    if run_PCA:
        limit_n_columns = False

    df_path = os.path.join(project_dir, patient_id, encounter_id, file_name)
    if not os.path.exists(df_path):
        warnings.warn(f"Skipping patient {patient_id} encounter {encounter_id} bc this files does not exist: {df_path}")
        return 0
    try:
        df = pd.read_csv(df_path, low_memory=False)
    except pd.errors.EmptyDataError:
        warnings.warn(f"Skipping patient {patient_id} encounter {encounter_id} bc this df is empty: {df_path}")
        return 0
    df['row_number'] = df.groupby('patient_id').cumcount()

    stay_duration = df["time"].iloc[-1]
    number_of_measurements = df["row_number"].iloc[-1]

    initial_keys = df.keys()

    nan_percentage_patient = {}
    for column in df.keys():
        nan_percentage_patient[column] = nan_percentage(df[column])

    FiO2_keys = []
    for key in list(df.keys()):
        if re.search("F[iI]O2", key) is not None:
            FiO2_keys.append(key)

    max_nan_percentage = 10
    columns_to_drop = [key for key, value in nan_percentage_patient.items() if value > max_nan_percentage]
    df = df.drop(columns_to_drop, axis=1).reset_index(drop=True)

    columns_not_interesting = ["Temperature C (calc)"]
    df_keys = df.keys()
    for column in columns_not_interesting:
        if column in df_keys:
            df = df.drop([column], axis=1).reset_index(drop=True)

    first_valid_indices = pd.DataFrame({col: df[col].first_valid_index() for col in df.columns},
                                       index=[0])

    first_valid_index = np.max(first_valid_indices)
    # print(first_valid_index)
    df = df.iloc[first_valid_index:].reset_index()
    # print(df.columns)
    string_columns = df.select_dtypes(include=['object']).keys()
    # df_patient = pd.get_dummies(df_patient, columns=string_columns)
    df = df.drop(string_columns, axis=1)

    min_change_times = None
    if limit_n_columns:
        columns_to_not_drop = ['patient_id', 'encounter_id']
        n_change_times = {}
        for column in df.keys():
            if column not in columns_to_not_drop:
                n_change_times[column] = count_changes(df[column])
                # print(f'{column}:{n_change_times[column]}')

        min_change_times = 5
        max_n_columns = 10
        while len(np.where(np.array(list(n_change_times.values())) > min_change_times)[0]) > max_n_columns:
            min_change_times +=1

        print(f"Final minimum number of change times is {min_change_times}")
        columns_to_drop = [key for key, value in n_change_times.items() if value < min_change_times]
        df = df.drop(columns_to_drop, axis=1).reset_index(drop=True)

    config = get_existing_config(project_dir, patient_id, encounter_id)

    config["stay_duration"] = stay_duration
    config["number_of_measurements"] = int(number_of_measurements)
    config["FiO2_keys"] = FiO2_keys
    config["analysis_parameters"]["max_nan_percentage"] = max_nan_percentage
    config["initial_keys"] = list(initial_keys)
    config["analysis_parameters"]["first_valid_index"] = int(first_valid_index)
    config["string_columns"] = list(string_columns)
    config["analysis_parameters"]["number_of_change_times"] = min_change_times
    config["hadm_id"] = int(df["hadm_id"].iloc[0])
    path_save = os.path.join(project_dir, patient_id, encounter_id, "saves")

    with open(os.path.join(path_save, f"config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # generate_patient_plot(df)
    df.to_csv(os.path.join(path_save, f"patient_mimic_format_preprocessed.csv"), index=False)


# @profile
def run_patient_encounter(patient_id, encounter_id, project_dir, file_name = "saves/patient_mimic_format_preprocessed.csv", plot=False, run_PCA=True):
    print(f"Starting patient : {patient_id}, encounter {encounter_id}")

    df_path = os.path.join(project_dir, patient_id, encounter_id, file_name)
    if not os.path.exists(df_path):
        warnings.warn(f"Skipping patient {patient_id} encounter {encounter_id} bc this files does not exist: {df_path}")
        return 0
    try:
        df = pd.read_csv(df_path, low_memory=False)
    except pd.errors.EmptyDataError:
        warnings.warn(f"Skipping patient {patient_id} encounter {encounter_id} bc this df is empty: {df_path}")
        return 0


    df_interest_start = max(30, int(len(df) * 0.01))
    window_length_fit = max(30, int(len(df) * 0.1))


    print(f"df_interest_start: {df_interest_start}, window_length: {window_length_fit}")

    non_medical_columns = ["index", "time", "patient_id", "row_number", "encounter_id", "hadm_id"]
    df_patient_medical_data = df.drop(non_medical_columns, axis=1)

    fit_data = df_patient_medical_data.iloc[df_interest_start:df_interest_start + window_length_fit]
    fit_keys = list(fit_data.keys())
    # print(fit_data.keys())
    scaler = StandardScaler()
    try:
        fit_data = scaler.fit_transform(fit_data)
    except ValueError:
        warnings.warn(f"Skipping patient {patient_id} encounter {encounter_id} bc this fit data is invalid")
        print(fit_data)
        return 0


    # PCA step
    n_pca_components = min(10, len(fit_keys))  # Set to 10 or the number of available features, whichever is smaller
    pca = PCA(n_components=n_pca_components)
    if run_PCA:    fit_data = pca.fit_transform(fit_data)

    path_plots = os.path.join(project_dir, patient_id, encounter_id, "plots")
    if os.path.exists(path_plots):
        shutil.rmtree(path_plots)
    os.mkdir(path_plots)


    plot_gif = False
    if plot_gif:
        batch_size = 30
        # Create a figure and axis
        data = pca.transform(scaler.transform(df_patient_medical_data))
        fig, ax = plt.subplots()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        scat = ax.scatter([], [], color='blue')
        x = data[:, 0]
        y = data[:, 1]

        def update(frame):
            fit_data_end = df_interest_start + window_length_fit
            if frame*batch_size < fit_data_end:
                ax.scatter(x[frame*batch_size:(frame+1)*batch_size],
                           y[frame*batch_size:(frame+1)*batch_size], color='blue', label="Fit data")
            else:
                ax.scatter(x[frame * batch_size:(frame + 1) * batch_size],
                           y[frame * batch_size:(frame + 1) * batch_size], color='red', label="Test data")
            return scat,
        ax.legend(loc='best')
        ani = FuncAnimation(fig, update, frames=len(x)//batch_size, interval=0, blit=True)
        ani.save(os.path.join(path_plots, "scatter.gif"), writer='imagemagick', fps=10)
        plt.close()

    n_components = 1
    covariance_type = "full"
    if run_PCA: covariance_type = "diag"

    gm = GaussianMixture(n_components=n_components,
                         random_state=0,
                         covariance_type=covariance_type).fit(fit_data)

    window_length = max(30, int(len(df_patient_medical_data) * 0.05))
    len_scores = len(df_patient_medical_data) - window_length - df_interest_start

    step = int(0.2*window_length)
    idx_where_to_compute_score = range(0, len_scores, step)
    scores = [0] * len(idx_where_to_compute_score)


    for i in idx_where_to_compute_score:
        data_to_score = df_patient_medical_data.iloc[df_interest_start + i:df_interest_start + i + window_length]
        data_to_score = scaler.transform(data_to_score)
        if run_PCA:
            data_to_score = pca.transform(data_to_score)
            scores[i // step] = my_score(data_to_score, gm)
        else:   scores[i // step] = gm.score(data_to_score)

        # scores[i//step] = gm.score(scaler.transform(
        #     df_patient_medical_data.iloc[df_interest_start + i:df_interest_start + i + window_length]))

    config = get_existing_config(project_dir, patient_id, encounter_id)
    hadm_id = config["hadm_id"]
    ARDS_onset_time = get_ARDS_onset_time(hadm_id, os.path.join(project_dir, patient_id, encounter_id), df["time"][0])

    if ARDS_onset_time is not None:
        plt.axvline(ARDS_onset_time, label="ARDS_onset_time", color="orange", linestyle="--")

    plt.plot(df["time"].iloc[idx_where_to_compute_score], scores, label="log_likelyhood")

    # plt.axvline(ards_onset_time, color="r", label="ARDS onset time")
    plt.xlabel("time (hrs)")
    plt.ylabel("log_likelyhood")
    plt.title(f"patient id: {patient_id}")
    plt.axvline(df["time"].iloc[df_interest_start], label="t_fit_start", color="green")
    plt.axvline(df["time"].iloc[df_interest_start + window_length_fit], label="t_fit_end", color="red")
    plt.legend()
    plt.savefig(os.path.join(project_dir, patient_id, encounter_id, "plots", f"scores.png"))
    if plot:    plt.show()
    else: plt.close()

    generate_patient_plot(df, patient_id, ARDS_onset_time, plot,
                          os.path.join(project_dir, patient_id, encounter_id, "plots", f"data.png"),
                          left_bar = df["time"].iloc[df_interest_start],
                          right_bar = df["time"].iloc[df_interest_start + window_length_fit])


    path_save = os.path.join(project_dir, patient_id, encounter_id, "saves")

    np.save(os.path.join(path_save, f"scores.npy"), scores)
    np.save(os.path.join(path_save, f"times.npy"), df["time"].iloc[idx_where_to_compute_score])




    config["patient_id"] = patient_id
    config["encounter_id"] = encounter_id
    config["fit_keys"] = fit_keys
    config["analysis_parameters"]["df_interest_start"] = df_interest_start
    config["analysis_parameters"]["window_length_fit"] = window_length_fit
    config["analysis_parameters"]["n_components"] = n_components
    if run_PCA:     config["analysis_parameters"]["method"] = "GMM with PCA"
    else: config["analysis_parameters"]["method"] = "GMM without PCA"
    config["analysis_parameters"]["window_length"] = window_length
    config["len_scores"] = len_scores

    with open(os.path.join(path_save, f"config.json"), "w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    for patient_id in os.listdir(project_dir):
        for encounter_id in os.listdir(os.path.join(project_dir, patient_id)):
            if os.path.exists(os.path.join(project_dir, patient_id, encounter_id, "saves", "patient_mimic_format_preprocessed.csv")):
                # print("Exists")
                shutil.move(os.path.join(project_dir, patient_id, encounter_id, "saves", "patient_mimic_format_preprocessed.csv"), os.path.join(project_dir, patient_id, encounter_id, "patient_mimic_format_preprocessed.csv"))
            # else:
            #     print("Does not exist")
            run_PCA = True
            run_patient_encounter_preprocessing(patient_id, encounter_id, project_dir, run_PCA=run_PCA)
            run_patient_encounter(patient_id, encounter_id, project_dir, plot=False, run_PCA=run_PCA)
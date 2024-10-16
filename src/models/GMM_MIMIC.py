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

def nan_percentage(column):
    return column.isna().mean() * 100


# Function to count changes for each column
def count_changes(df):
    return (df != df.shift()).sum()


def my_score(X, gm):
    scores = np.zeros(X.shape)
    std = np.sqrt(gm.covariances_)
    for sample in range(X.shape[0]):
        for j in range(X.shape[1]):
            scores[sample, j] = (X[sample, j] / np.mean(std[:, j])) ** 2
    return np.mean(scores)


def get_ARDS_onset_time(hadm_id, save_path, start_time):
    patient_dir = save_path
    ARDS_onset_time_txt = get_LLM_result(patient)["onset_timestamp"]
    if ARDS_onset_time_txt[0] == "-":
        return None
    else:
        try:
            ARDS_onset_time = int(ARDS_onset_time_txt.split(" ")[0]) * 24 + int(ARDS_onset_time_txt.split(" ")[-1])
            return ARDS_onset_time
        except ValueError as e:
            print(f"Error: {e}")
            return -1


def generate_patient_plot(patient, ARDS_onset_time, show, left_bar=None, right_bar=None):
    # print(f"ARDS_onset_time: {ARDS_onset_time}")
    # print("left_bar:", left_bar)
    # print("right_bar:", right_bar)

    df = patient.processed_df
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
        ax1.axvline(ARDS_onset_time, label="ARDS_onset_time", color="orange", linestyle="--")

    ax1.legend(loc='upper left')

    # fig.tight_layout()
    plt.title(f"Data for patient {patient.subject_id}")
    # print(save_path)
    plt.savefig(os.path.join(patient.save_path, "plots", "data.png"))
    if show: plt.show()
    plt.close()
    # return ax1


def run_patient_preprocessing(patient:Patient, run_PCA=False, limit_n_columns=False, plot=False, columns_to_keep=None, min_len_patient=0):

    df = patient.raw_df.copy()
    if len(df) < min_len_patient:
        patient.del_patient(f"df has len < {min_len_patient}")
        return -1
    # print(list(df.columns))
    # print(columns_to_keep)
    columns_to_keep_in_df = []
    for key in columns_to_keep:
        if key in df.columns:
            columns_to_keep_in_df.append(key)

    # print(columns_to_keep_in_df)
    # print(list(df.columns))

    if columns_to_keep is not None:
        df = df[columns_to_keep_in_df + ["time"]]
        df = df.drop_duplicates(subset=columns_to_keep_in_df, keep="first")
    # df['row_number'] = df.cumcount()

    stay_duration = df["time"].iloc[-1]
    number_of_measurements = len(df)

    initial_keys = df.keys()

    nan_percentage_patient = {}
    for column in df.keys():
        nan_percentage_patient[column] = nan_percentage(df[column])

    FiO2_keys = []
    for key in list(df.keys()):
        if re.search("F[iI]O2", key) is not None:
            FiO2_keys.append(key)

    max_nan_percentage = 100
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
    df = df.iloc[first_valid_index:].reset_index()
    string_columns = df.select_dtypes(include=['object']).keys()
    df = df.drop(string_columns, axis=1)

    min_change_times = None
    if limit_n_columns:
        columns_to_not_drop = ['patient_id', 'encounter_id']
        n_change_times = {}
        for column in df.keys():
            if column not in columns_to_not_drop:
                n_change_times[column] = count_changes(df[column])

        min_change_times = 5
        max_n_columns = 10
        while len(np.where(np.array(list(n_change_times.values())) > min_change_times)[0]) > max_n_columns:
            min_change_times += 1

        # print(f"Final minimum number of change times is {min_change_times}")
        columns_to_drop = [key for key, value in n_change_times.items() if value < min_change_times]
        df = df.drop(columns_to_drop, axis=1).reset_index(drop=True)

    config = patient.get_existing_config()

    config["stay_duration"] = stay_duration
    config["number_of_measurements"] = int(number_of_measurements)
    config["FiO2_keys"] = FiO2_keys
    config["analysis_parameters"]["run_pca"] = run_PCA
    config["analysis_parameters"]["max_nan_percentage"] = max_nan_percentage
    config["initial_keys"] = list(initial_keys)
    config["analysis_parameters"]["first_valid_index"] = int(first_valid_index)
    config["string_columns"] = list(string_columns)
    config["analysis_parameters"]["number_of_change_times"] = min_change_times
    # print(df)
    patient.save_config(config)
    # print(df.columns)
    patient.save_processed_df(df)


# @profile
def run_patient_encounter(patient, plot=False, run_PCA=True, min_len_df = 0):
    # print(f"Starting patient : {patient.subject_id}, encounter {patient.hadm_id}")
    try:
        df = patient.get_processed_df()
    except FileNotFoundError:
        patient.del_patient(reason = "processed df was not found")
        return -1

    if len(df) < min_len_df:
        patient.del_patient(reason = f"len df < {min_len_df}")
        return -1
    try:
        # df_interest_start = int(df["time"].iloc[0])
        df_interest_start = 0
    except KeyError:
        patient.del_patient("no time key in df")
        return -1

    window_length_fit = max(1, int(len(df) * 0.05))
    # print(f"df_interest_start: {df_interest_start}, window_length: {window_length_fit}")

    non_medical_columns = ["index", "time"]
    df_patient_medical_data = df.drop(non_medical_columns, axis=1)

    fit_data = df_patient_medical_data.iloc[df_interest_start:df_interest_start + window_length_fit]
    fit_keys = list(fit_data.keys())

    scaler = StandardScaler()
    try:
        fit_data = scaler.fit_transform(fit_data)
    except ValueError:
        patient.del_patient("fit data is invalid")
        return 0

    df_final = pd.DataFrame(scaler.transform(df_patient_medical_data.copy()), columns=fit_keys)
    df_final["time"] = df["time"]
    patient.save_final_df(df_final)

    # PCA step
    n_pca_components = min(10, len(fit_keys))  # Set to 10 or the number of available features, whichever is smaller
    pca = PCA(n_components=n_pca_components)
    if run_PCA:    fit_data = pca.fit_transform(fit_data)

    path_plots = os.path.join(patient.save_path, "plots")
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
            if frame * batch_size < fit_data_end:
                ax.scatter(x[frame * batch_size:(frame + 1) * batch_size],
                           y[frame * batch_size:(frame + 1) * batch_size], color='blue', label="Fit data")
            else:
                ax.scatter(x[frame * batch_size:(frame + 1) * batch_size],
                           y[frame * batch_size:(frame + 1) * batch_size], color='red', label="Test data")
            return scat,

        ax.legend(loc='best')
        ani = FuncAnimation(fig, update, frames=len(x) // batch_size, interval=0, blit=True)
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

    step = int(0.2 * window_length)
    idx_where_to_compute_score = range(0, len_scores, step)
    time_of_scores = [0] * len(idx_where_to_compute_score)
    scores = [0] * len(idx_where_to_compute_score)

    for i in idx_where_to_compute_score:
        data_to_score = df_patient_medical_data.iloc[df_interest_start + i:df_interest_start + i + window_length]
        data_to_score = scaler.transform(data_to_score)
        time_of_scores[i // step] = df["time"].iloc[df_interest_start + i+window_length//2]
        if run_PCA:
            data_to_score = pca.transform(data_to_score)
            scores[i // step] = my_score(data_to_score, gm)
        else:
            scores[i // step] = gm.score(data_to_score)

        # scores[i//step] = gm.score(scaler.transform(
        #     df_patient_medical_data.iloc[df_interest_start + i:df_interest_start + i + window_length]))
    hadm_id = patient.hadm_id

    # ARDS_onset_time = get_ARDS_onset_time(hadm_id, patient.save_path, patient.time_start)
    ARDS_onset_time = None
    # print(min(df["time"]))
    # print(f"Actual start time: ", df["time"].iloc[0])
    # print(f"Fit start time: ", df["time"].iloc[df_interest_start])
    # print(f"Fit stop time: ", df["time"].iloc[df_interest_start + window_length_fit])
    # print(f"ARDS time:", ARDS_onset_time)

    config = patient.get_existing_config()

    # if df["time"].iloc[df_interest_start + window_length_fit] > ARDS_onset_time:
    #     print(f"ARDS onset is during fit time.")
    # else:
    #     print(f"ARDS onset is not during fit time.")

    if ARDS_onset_time is not None:
        plt.axvline(ARDS_onset_time, label="ARDS_onset_time", color="orange", linestyle="--")

    plt.plot(df["time"].iloc[idx_where_to_compute_score], scores, label="log_likelyhood")
    # print(f"len df {len(df)}, df interest start {df_interest_start}, window length fit {window_length_fit}")
    # plt.axvline(ards_onset_time, color="r", label="ARDS onset time")
    plt.xlabel("time (hrs)")
    plt.ylabel("log_likelyhood")
    plt.title(f"patient id: {patient.subject_id}")
    plt.axvline(df["time"].iloc[df_interest_start], label="t_fit_start", color="green")
    plt.axvline(df["time"].iloc[df_interest_start + window_length_fit], label="t_fit_end", color="red")
    plt.legend()
    plt.savefig(os.path.join(patient.save_path, "plots", f"scores.png"))

    if plot:
        plt.show()
    else:
        plt.close()

    generate_patient_plot(patient, ARDS_onset_time, plot,
                          left_bar=df["time"].iloc[df_interest_start],
                          right_bar=df["time"].iloc[df_interest_start + window_length_fit])

    patient.save_scores(scores)
    patient.save_times(time_of_scores)

    # np.save(os.path.join(patient.save_path, f"scores.npy"), scores)
    # np.save(os.path.join(patient.save_path, f"times.npy"), df["time"].iloc[idx_where_to_compute_score])

    config["patient_id"] = patient.subject_id
    config["encounter_id"] = patient.hadm_id
    config["fit_keys"] = fit_keys
    config["analysis_parameters"]["df_interest_start"] = df_interest_start
    config["analysis_parameters"]["window_length_fit"] = window_length_fit
    config["analysis_parameters"]["n_components"] = n_components
    if run_PCA:
        config["analysis_parameters"]["method"] = "models with PCA"
    else:
        config["analysis_parameters"]["method"] = "models without PCA"
    config["analysis_parameters"]["window_length"] = window_length
    config["len_scores"] = len_scores
    config["ards_onset_time"] = ARDS_onset_time
    patient.save_config(config)


# Define a function to process each patient
def process_patient(row, project_dir, columns_to_keep):
    # print(row["subject_id"])
    patient = Patient.load(project_dir, str(row["subject_id"]), str(row["hadm_id"]))
    run_PCA = False
    run_patient_preprocessing(patient, run_PCA=run_PCA, columns_to_keep=columns_to_keep)
    # run_patient_encounter(patient, plot=False, run_PCA=run_PCA)
    # print()

def main():
    project_dir = "/home/julien/Documents/stage/data/MIMIC/mini_test"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))
    columns_to_keep = ["Heart Rate", "SpO2", "Respiratory Rate", "NBP Mean", "NBP [Systolic]", "NBP [Diastolic]",
                       "Temperature F", 'PEEP Set', 'FiO2 Set', 'Arterial PaO2']

    t = time.time()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     [executor.submit(process_patient, row, project_dir, columns_to_keep) for _, row in patients_list_df.iterrows()]

    for index, row in patients_list_df.iterrows():
        # print(row["subject_id"])
        patient = Patient.load(project_dir, str(row["subject_id"]), str(row["hadm_id"]))
        run_PCA = False
        run_patient_preprocessing(patient, run_PCA=run_PCA, columns_to_keep=columns_to_keep)
        # run_patient_encounter(patient, plot=False, run_PCA=run_PCA)
        # print()
    print(f"Total time : {time.time() - t}")

if __name__ == "__main__":
    project_dir = "/home/julien/Documents/stage/data/MIMIC/final"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))
    columns_to_keep = ["Heart Rate", "SpO2", "Respiratory Rate", "Arterial BP [Systolic]", "Arterial BP [Diastolic]",
                             "Temperature F", 'PEEP Set', 'FiO2 Set', 'Arterial PaO2']
    t = time.time()
    # print(os.path.join(project_dir, "patients.csv"))
    # patients_list_df = patients_list_df.iloc[:400]
    # patients_list_df.to_csv(os.path.join(project_dir, "patients.csv"), index=False)
    for index, row in patients_list_df.iterrows():
        if index%100 == 0:
            print(index)
        patient = Patient.load(project_dir, str(int(row["subject_id"])), str(int(row["hadm_id"])))
        if patient is not None:
            run_PCA = False
            run_patient_preprocessing(patient, run_PCA=run_PCA, columns_to_keep=columns_to_keep)
            # run_patient_encounter(patient, plot=False, run_PCA=run_PCA)



        # if __name__ == "__main__":
        #     cProfile.run('main()', 'profile_output')
        #     p = pstats.Stats('profile_output')
        #     p.strip_dirs().sort_stats('time').print_stats(20)  # Print top 20 time-consuming lines
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

def cusum(times, log_likelihoods, threshold=10, drift=0):
    """
    Apply the CUSUM algorithm to detect change points in the log-likelihoods.

    Parameters:
    - log_likelihoods: Array or list of log-likelihood values.
    - threshold: Threshold value for detecting significant changes.
    - drift: Allowable drift before declaring a change point (default is 0).

    Returns:
    - change_points: Indices of detected change points.
    - S_pos: The positive cumulative sums.
    - S_neg: The negative cumulative sums.
    """
    # Initialize variables
    S_pos, S_neg = np.zeros(len(log_likelihoods)), np.zeros(len(log_likelihoods))
    change_points = []

    # Iterate through log-likelihood values to compute cumulative sums
    for i in range(1, len(log_likelihoods)):
        # Cumulative sums
        S_pos[i] = max(0, S_pos[i - 1] + log_likelihoods[i] - drift)
        S_neg[i] = max(0, S_neg[i - 1] - log_likelihoods[i] - drift)

        # Check if either cumulative sum exceeds the threshold
        if S_pos[i] > threshold:
            change_points.append(i)
            S_pos[i] = 0  # Reset the positive cumulative sum after a change point

        if S_neg[i] < -threshold:
            change_points.append(i)
            S_neg[i] = 0  # Reset the negative cumulative sum after a change point

    return change_points, S_pos, S_neg

def plot_cusum(times, log_likelihoods, change_points, S_pos, S_neg, patient=None):
    fig, ax1 = plt.subplots()
    # plt.figure(figsize=(10, 6))
    ax1.plot(times, log_likelihoods, label='Log-Likelihoods')
    # ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    # ax2.plot(times[1:], np.log(S_neg[1:])-np.log(S_neg[:-1]), label='S_neg (CUSUM) derivative', linestyle='--')
    # ax2.plot(times, S_pos, label='S_pos (CUSUM)', linestyle='--')
    ax2.plot(times, S_neg, label='S_neg (CUSUM)', linestyle='--')
    # for cp in change_points:
    #     plt.axvline(times[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else '')

    plt.title('CUSUM Change Point Detection')
    if patient is not None:
        plt.title(f'CUSUM Change Point Detection, patient {patient}')
    plt.xlabel('Time Index')
    plt.ylabel('Log-Likelihood / CUSUM Value')
    alpha = 1e3

    # first_index_over_alpha = np.argwhere(np.log(S_neg[1:])-np.log(S_neg[:-1]) > alpha)
    first_index_over_alpha = np.argwhere(S_neg > alpha)

    if len(first_index_over_alpha) > 0:
        print(S_pos[first_index_over_alpha[0]], S_neg[first_index_over_alpha[0]])

        plt.axvline(times[first_index_over_alpha][0], color='red', linestyle='--')
    plt.legend()
    plt.show()


save_path = "/home/julien/Documents/stage/data/MIMIC/cohorts/saves"
for patient in os.listdir(save_path):
    scaler = StandardScaler()
    fit_length = np.load(os.path.join(save_path, patient, "fit_length.npy"))*2
    patient_likelihoods = np.load(os.path.join(save_path, patient, "scores.npy"))
    # patient_likelihoods = np.log(-patient_likelihoods-np.max(patient_likelihoods)-1)
    scaler.fit(patient_likelihoods[:fit_length].reshape(-1, 1))

    patient_likelihoods = scaler.transform(patient_likelihoods.reshape(-1, 1))
    # print(patient_likelihoods)
    patient_times = np.load(os.path.join(save_path, patient, "times.npy"))
    # plt.plot(patient_times, patient_likelihoods)
    change_points, S_pos, S_neg = cusum(patient_times, patient_likelihoods)
    # print(change_points)
    # print(S_pos, S_neg)
    plot_cusum(patient_times, patient_likelihoods, change_points, S_pos, S_neg, patient=patient)
    # break/

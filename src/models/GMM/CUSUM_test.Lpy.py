import numpy as np
import matplotlib.pyplot as plt

def cusum(log_likelihoods, threshold=5, drift=0):
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
        S_neg[i] = min(0, S_neg[i - 1] + log_likelihoods[i] - drift)

        # Check if either cumulative sum exceeds the threshold
        if S_pos[i] > threshold:
            change_points.append(i)
            S_pos[i] = 0  # Reset the positive cumulative sum after a change point

        if S_neg[i] < -threshold:
            change_points.append(i)
            S_neg[i] = 0  # Reset the negative cumulative sum after a change point

    return change_points, S_pos, S_neg


# Example usage
log_likelihoods = np.random.normal(0, 1, 100)  # Simulated log-likelihoods

# Detect change points
change_points, S_pos, S_neg = cusum(log_likelihoods, threshold=5, drift=0)

print("Detected Change Points:", change_points)
# Plot the log-likelihoods and detected change points
plt.figure(figsize=(10, 6))
plt.plot(log_likelihoods, label='Log-Likelihoods')
plt.plot(S_pos, label='S_pos (CUSUM)', linestyle='--')
plt.plot(S_neg, label='S_neg (CUSUM)', linestyle='--')

for cp in change_points:
    plt.axvline(cp, color='red', linestyle='--', label='Change Point' if cp == change_points[0] else '')

plt.title('CUSUM Change Point Detection')
plt.xlabel('Time Index')
plt.ylabel('Log-Likelihood / CUSUM Value')
plt.legend()
plt.show()
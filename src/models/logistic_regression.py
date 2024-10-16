import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
from multiprocessing import Pool

# Assuming the Patient class is defined in src.utils.patient
from src.utils.patient import Patient


# Function to get processed patient data as difference vectors
def to_diff(df, mode="concat"):
    df_diff = df.diff().iloc[1:]
    df_no_time = df.drop(columns='time').iloc[1:]
    df_diff_no_time = df_diff.drop(columns='time')
    if mode == "concat":
        return pd.concat([df_no_time, df_diff_no_time.div(df_diff['time'].values, axis=0)], axis=1)
    return df_no_time


def load_all_patients(hadm_ids, patient_list_df, project_dir):
    patients_data = {}
    for hadm_id in hadm_ids:
        subject_id = patient_list_df.loc[patient_list_df["hadm_id"] == hadm_id, "subject_id"].values[0]
        patient = Patient.load(project_dir, str(subject_id), str(hadm_id))
        patients_data[hadm_id] = {
            'patient': patient,
            'config': patient.get_existing_config(),
            'df_final': patient.get_processed_df()
        }
    return patients_data


def process_patient_samples(args):
    hadm_id, data, time_window = args
    positive_samples = []
    negative_samples = []
    ards_onset_time = float(data['config']["proxy_label"])
    df_final = data['df_final']

    times = df_final['time'].values[:-1]
    time_plus_window = times + time_window

    negative_mask = (time_plus_window < ards_onset_time)
    positive_mask = (times < ards_onset_time) & (time_plus_window > ards_onset_time)

    indices = np.arange(len(times))
    negative_indices = indices[negative_mask]
    positive_indices = indices[positive_mask]

    # Compute differences once
    df_diff = df_final.diff().iloc[1:]
    df_no_time = df_final.drop(columns='time').iloc[1:]
    df_diff_no_time = df_diff.drop(columns='time')
    diff_vectors = pd.concat([df_no_time, df_diff_no_time.div(df_diff['time'].values, axis=0)], axis=1).values

    # Collect negative samples
    negative_samples.extend(diff_vectors[negative_indices])
    # Collect positive samples
    positive_samples.extend(diff_vectors[positive_indices])

    return negative_samples, positive_samples


def extract_samples_parallel(hadm_ids, patients_data, time_window):
    with Pool() as pool:
        args = [(hadm_id, patients_data[hadm_id], time_window) for hadm_id in hadm_ids]
        results = pool.map(process_patient_samples, args)

    negative_samples = []
    positive_samples = []
    for neg_samples, pos_samples in results:
        negative_samples.extend(neg_samples)
        positive_samples.extend(pos_samples)

    negative_samples = np.array(negative_samples)
    positive_samples = np.array(positive_samples)

    return negative_samples, positive_samples


def preprocess_data(X_train=None, X_test=None, imputer=None, scaler=None):
    # Impute missing values
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        if X_train is not None:
            X_train_imputed = imputer.fit_transform(X_train)
        else:
            X_train_imputed = None
    else:
        if X_train is not None:
            X_train_imputed = imputer.transform(X_train)
        else:
            X_train_imputed = None

    if X_test is not None:
        X_test_imputed = imputer.transform(X_test)
    else:
        X_test_imputed = None

    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        if X_train_imputed is not None:
            X_train_scaled = scaler.fit_transform(X_train_imputed)
        else:
            X_train_scaled = None
    else:
        if X_train_imputed is not None:
            X_train_scaled = scaler.transform(X_train_imputed)
        else:
            X_train_scaled = None

    if X_test_imputed is not None:
        X_test_scaled = scaler.transform(X_test_imputed)
    else:
        X_test_scaled = None

    return X_train_scaled, X_test_scaled, imputer, scaler


# Function to train the XGBoost classifier with preprocessing
def run_xgboost_classifier(train_subjects, patient_list_df, time_window, project_dir):
    # Load all patient data
    patients_data = load_all_patients(train_subjects, patient_list_df, project_dir)

    # Extract samples using the new function
    negative_samples_train, positive_samples_train = extract_samples_parallel(
        train_subjects, patients_data, time_window
    )

    X = np.vstack((negative_samples_train, positive_samples_train))
    y = np.hstack((np.zeros(negative_samples_train.shape[0]), np.ones(positive_samples_train.shape[0])))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    X_train_preprocessed, X_val_preprocessed, imputer, scaler = preprocess_data(X_train, X_val)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

    # Print class distribution after SMOTE
    print('Resampled dataset shape %s' % Counter(y_train_resampled))

    # Define the XGBoost classifier with optimized parameters
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        objective='binary:logistic',
        n_estimators=500,  # Reduced from 10000
        max_depth=10,
        learning_rate=0.1,
        early_stopping_rounds=40,
        verbosity=0  # Set verbosity to 0 to disable printing
    )

    # Train the model with early stopping on validation set
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val_preprocessed, y_val)],
        verbose=False
    )

    return model, imputer, scaler


# Function to test the model and plot the ROC AUC curve
def xgboost_classifier_test(test_subjects, patient_list_df, model, imputer, scaler, time_window, project_dir):
    # Load all patient data
    patients_data = load_all_patients(test_subjects, patient_list_df, project_dir)

    # Extract samples using the new function
    negative_samples_test, positive_samples_test = extract_samples_parallel(
        test_subjects, patients_data, time_window
    )

    # Ensure that the arrays have the same number of dimensions (should be 2D)
    if negative_samples_test.ndim != 2 or positive_samples_test.ndim != 2:
        raise ValueError("Both negative and positive samples must be 2D arrays.")

    # Stack the negative and positive samples for testing
    X_test = np.vstack((negative_samples_test, positive_samples_test))
    y_test = np.hstack((np.zeros(len(negative_samples_test)), np.ones(len(positive_samples_test))))

    # Preprocess the test data using the imputer and scaler from training
    _, X_test_preprocessed, _, _ = preprocess_data(X_train=None, X_test=X_test, imputer=imputer, scaler=scaler)

    # Make predictions using the trained model
    y_test_pred = model.predict(X_test_preprocessed)

    # Print detailed classification report for the test set
    print("Testing results:")
    print(classification_report(y_test, y_test_pred))

    # Plot ROC AUC curve
    plot_roc_auc(model, X_test_preprocessed, y_test)


# Function to plot ROC AUC curve
def plot_roc_auc(model, X_test, y_test):
    # Get the predicted probabilities for class 1 (positive class)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

    # Compute AUC score
    auc_score = roc_auc_score(y_test, y_test_pred_proba)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


# Function to extract samples and times for a single patient
def extract_samples_for_patient(hadm_id, patient_data, time_window):
    samples = []
    times = []

    data = patient_data
    ards_onset_time = float(data['config']["proxy_label"])
    df_final = data['df_final']

    times_array = df_final['time'].values[:-1]
    indices = np.arange(len(times_array))

    # Compute differences once
    df_diff = df_final.diff().iloc[1:]
    df_no_time = df_final.drop(columns='time').iloc[1:]
    df_diff_no_time = df_diff.drop(columns='time')
    diff_vectors = pd.concat([df_no_time, df_diff_no_time.div(df_diff['time'].values, axis=0)], axis=1).values

    # Collect all samples and times
    samples.extend(diff_vectors)
    times.extend(times_array)

    samples = np.array(samples)
    times = np.array(times)

    return samples, times, ards_onset_time


# Function to plot prediction probabilities over time for one patient
def plot_prediction_probabilities_for_patient(hadm_id, patient_list_df, model, imputer, scaler, time_window,
                                              project_dir):
    # Load patient data
    patients_data = load_all_patients([hadm_id], patient_list_df, project_dir)
    patient_data = patients_data[hadm_id]

    # Extract samples and times for the patient
    samples, times, ards_onset_time = extract_samples_for_patient(
        hadm_id, patient_data, time_window
    )

    # Preprocess the data
    _, X_preprocessed, _, _ = preprocess_data(X_train=None, X_test=samples, imputer=imputer, scaler=scaler)

    # Get the predicted probabilities for class 1 (positive class)
    y_pred_proba = model.predict_proba(X_preprocessed)[:, 1]

    # Debugging prints
    print(f"Number of times: {len(times)}")
    print(f"Number of samples: {samples.shape[0]}")
    print(f"Number of predictions: {len(y_pred_proba)}")

    # Sort times and predictions together for plotting
    sorted_indices = np.argsort(times)

    times_sorted = times[sorted_indices]
    y_pred_proba_sorted = y_pred_proba[sorted_indices]

    # Plot prediction probabilities over time
    plt.figure(figsize=(10, 6))
    plt.plot(times_sorted, y_pred_proba_sorted, label="Prediction Probability")
    plt.axvline(x=ards_onset_time, color='red', linestyle='--', label='ARDS Onset')
    plt.xlabel('Time')
    plt.ylabel('Prediction Probability')
    plt.title(f'Prediction Probability Over Time for Patient {hadm_id}')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    time_window = 12  # Time window before/after ARDS onset

    project_dir = "/home/julien/Documents/stage/data/MIMIC/final"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patient_ARDS_df.csv"))

    cohort = patients_list_df['hadm_id']
    train_subjects, test_subjects = train_test_split(cohort, test_size=0.2, random_state=0)

    print(f"Cohort loaded with {len(train_subjects)} training subjects and {len(test_subjects)} test subjects")

    # Train the XGBoost model
    model, imputer, scaler = run_xgboost_classifier(
        train_subjects, patients_list_df, time_window, project_dir
    )

    # Test the model and plot ROC AUC curve
    xgboost_classifier_test(
        test_subjects, patients_list_df, model, imputer, scaler, time_window, project_dir
    )

    for i in range(100):
        # Plot prediction probabilities over time for one patient
        hadm_id_to_test = test_subjects.iloc[i]  # Choose the first patient from the test set
        plot_prediction_probabilities_for_patient(
            hadm_id_to_test, patients_list_df, model, imputer, scaler, time_window, project_dir
        )


if __name__ == "__main__":
    main()

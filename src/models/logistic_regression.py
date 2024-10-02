import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from collections import Counter
import os
from src.utils.patient import Patient

# Function to get processed patient data as difference vectors
def to_diff(df, mode="concat"):
    df_diff = df.diff()[1:]
    if mode == "concat":
        return pd.concat([df.drop("time", axis=1)[1:], df_diff.drop("time", axis=1).div(df_diff.time, axis=0)], axis=1)
    return df.drop("time", axis=1)

# Function to get samples with a time window relative to ARDS onset
def get_samples_with_time_window(subjects, patient_list_df, time_window):
    project_dir = "/home/julien/Documents/stage/data/MIMIC/full"
    positive_samples = []
    negative_samples = []

    for subject_id in subjects:
        hadm_id = patient_list_df[patient_list_df["subject_id"] == subject_id]["hadm_id"].values[0]
        patient = Patient.load(project_dir, str(subject_id), str(hadm_id))
        config = patient.get_existing_config()
        ards_onset_time = int(config["proxy_label"])
        df_final = patient.get_processed_df()

        for i in range(len(df_final) - 1):
            time = df_final.iloc[i]["time"]
            time_plus_window = time + time_window

            # Negative samples: both time and time + window are before ARDS onset
            if time_plus_window < ards_onset_time:
                diff_vector = to_diff(df_final.iloc[i:i+2])
                negative_samples.append(diff_vector.values.flatten())  # Flatten the array into 1D

            # Positive samples: time is before ARDS onset, but time + window is after onset
            elif time < ards_onset_time and time_plus_window > ards_onset_time:
                diff_vector = to_diff(df_final.iloc[i:i+2])
                positive_samples.append(diff_vector.values.flatten())  # Flatten the array into 1D

    return np.array(negative_samples), np.array(positive_samples)

# Function to scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Function to impute missing values
def impute_missing_values(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed, imputer

# Function to train the XGBoost classifier
def run_xgboost_classifier(negative_samples, positive_samples):
    X = np.vstack((negative_samples, positive_samples))
    y = np.hstack((np.zeros(negative_samples.shape[0]), np.ones(positive_samples.shape[0])))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values (replace NaNs with the mean of the column)
    X_train, X_val, imputer = impute_missing_values(X_train, X_val)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Print class distribution after SMOTE
    print('Resampled dataset shape %s' % Counter(y_train))

    # Feature scaling
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)

    # Define the XGBoost classifier
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        objective='binary:logistic',
        n_estimators=2300,
        max_depth=5,
        learning_rate=0.1
    )

    # Train the model with early stopping on validation set
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],  # Validation set for early stopping
        verbose=True
    )

    return model, imputer, scaler

# Function to test the model and plot the ROC AUC curve
def test_xgboost_classifier(subjects, patient_list_df, model, imputer, scaler, time_window):
    project_dir = "/home/julien/Documents/stage/data/MIMIC/full"
    positive_samples = []
    negative_samples = []

    for subject_id in subjects:
        hadm_id = patient_list_df[patient_list_df["subject_id"] == subject_id]["hadm_id"].values[0]
        patient = Patient.load(project_dir, str(subject_id), str(hadm_id))
        config = patient.get_existing_config()
        ards_onset_time = int(config["proxy_label"])
        df_final = patient.get_processed_df()

        for i in range(len(df_final) - 1):
            time = df_final.iloc[i]["time"]
            time_plus_window = time + time_window

            # Negative samples: both time and time + window are before ARDS onset
            if time_plus_window < ards_onset_time:
                diff_vector = to_diff(df_final.iloc[i:i+2])
                negative_samples.append(diff_vector.values.flatten())  # Flatten to ensure 2D

            # Positive samples: time is before ARDS onset, but time + window is after onset
            elif time < ards_onset_time and time_plus_window > ards_onset_time:
                diff_vector = to_diff(df_final.iloc[i:i+2])
                positive_samples.append(diff_vector.values.flatten())  # Flatten to ensure 2D

    # Convert lists of samples to numpy arrays
    negative_samples = np.array(negative_samples)
    positive_samples = np.array(positive_samples)

    # Ensure that the arrays have the same number of dimensions (should be 2D)
    if negative_samples.ndim != 2 or positive_samples.ndim != 2:
        raise ValueError("Both negative and positive samples must be 2D arrays.")

    # Stack the negative and positive samples for testing
    X_test = np.vstack((negative_samples, positive_samples))
    y_test = np.hstack((np.zeros(len(negative_samples)), np.ones(len(positive_samples))))

    # Impute missing values using the imputer from training
    X_test_imputed = imputer.transform(X_test)

    # Scale features using the scaler from training
    X_test_scaled = scaler.transform(X_test_imputed)

    # Make predictions using the trained model
    y_test_pred = model.predict(X_test_scaled)

    # Print detailed classification report for the test set
    print("Testing results:")
    print(classification_report(y_test, y_test_pred))

    # Plot ROC AUC curve
    plot_roc_auc(model, X_test_scaled, y_test)

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

if __name__ == "__main__":
    time_window = 12  # Time window before/after ARDS onset

    project_dir = "/home/julien/Documents/stage/data/MIMIC/full"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patient_ARDS_df.csv"))

    cohort = pd.read_csv("cohort.csv")['subject_id'].values
    train_subjects, test_subjects = train_test_split(cohort, test_size=0.2, random_state=0)

    print(f"Cohort loaded with {len(train_subjects)} training subjects and {len(test_subjects)} test subjects")

    # Get training samples
    negative_samples_train, positive_samples_train = get_samples_with_time_window(train_subjects, patients_list_df, time_window)

    # Train the XGBoost model
    model, imputer, scaler = run_xgboost_classifier(negative_samples_train, positive_samples_train)

    # Test the model and plot ROC AUC curve
    test_xgboost_classifier(test_subjects, patients_list_df, model, imputer, scaler, time_window)

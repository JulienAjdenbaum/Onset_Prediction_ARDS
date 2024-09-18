import json
import os
import pandas as pd
from src.utils.db_utils import run_query
import datetime
import warnings
import numpy as np
# /home/julien/Documents/stage/data/MIMIC/cohorts_new/20587/100108/data/raw_df.csv

def timestamp_to_string(timestamp):
    """
    Convert a timestamp (datetime object) to a string in ISO format.
    """
    return timestamp.isoformat()


def string_to_timestamp(timestamp_str):
    """
    Convert a string in ISO format back to a datetime object.
    """
    timestamp_str = timestamp_str.values[0]
    return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S") if timestamp_str else None


def get_diagnoses(hadm_id):
    query = """
        SELECT long_title
        FROM DIAGNOSES_ICD
        JOIN D_ICD_DIAGNOSES ON DIAGNOSES_ICD.icd9_code = D_ICD_DIAGNOSES.icd9_code
        WHERE hadm_id = %(hadm_id)s
        """
    return list(run_query(query, {"hadm_id": int(hadm_id)}))


class Patient:
    def __init__(self, df: pd.DataFrame, root_path, is_load=False):
        self.subject_id = int(df.iloc[0]["subject_id"])
        self.hadm_id = int(df.iloc[0]["hadm_id"])
        self.main_path = os.path.join(root_path, str(self.subject_id), str(self.hadm_id))
        self.save_path = os.path.join(self.main_path, "data")
        self.diagnoses = get_diagnoses(self.hadm_id)
        self.processed_df = None

        if not os.path.exists(os.path.join(root_path, str(self.subject_id))):
            os.makedirs(os.path.join(root_path, str(self.subject_id)))
        if not os.path.exists(self.main_path):
            os.makedirs(self.main_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if is_load:
            self.raw_df = pd.read_csv(os.path.join(self.save_path, "raw_df.csv"), low_memory=False)
            self.time_start = df["time_start"]
            if os.path.exists(os.path.join(self.save_path, "processed_df.csv")):
                self.processed_df = pd.read_csv(os.path.join(self.save_path, "processed_df.csv"))
            return

        self.time_start = df["charttime"].min()

        rows = []

        for time in np.unique(df["charttime"]):
            time_feature = (time - self.time_start).total_seconds() / 3600
            rows.append(({"patient_id": self.subject_id, "hadm_id": self.hadm_id,
                          "time": time_feature, label: value}))
            rows_list = df[df["charttime"] == time]

            for index, row in rows_list.iterrows():
                time_feature = (row["charttime"] - self.time_start).total_seconds() / 3600
                label = row["label"]
                value = row["value"]


        df = pd.DataFrame(rows).sort_values(by="time").reset_index(drop=True)
        self.raw_df = df.ffill()
        self.save_raw_df()
        self.save_infos()
        self.add_patient_to_global_df(root_path)

    def save_infos(self):
        """
        Save the patient's data to the specified directory.
        """

        # Save other patient information as JSON
        patient_info = {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "diagnoses": self.diagnoses,
            "time_start": timestamp_to_string(self.time_start)
        }

        with open(os.path.join(self.main_path, "patient_info.json"), "w") as f:
            json.dump(patient_info, f, indent=4)

    def save_raw_df(self):
        self.raw_df.to_csv(os.path.join(self.save_path, "raw_df.csv"), index=False)

    def save_processed_df(self, df_processed=None):
        if df_processed is not None:
            self.processed_df = df_processed
        self.processed_df.to_csv(os.path.join(self.save_path, "processed_df.csv"), index=False)

    def add_patient_to_global_df(self, root_path):
        csv_path = os.path.join(root_path, "patients.csv")
        patient_dic = {"subject_id": self.subject_id,
                       "hadm_id": self.hadm_id}
        if not os.path.exists(csv_path):
            pd.DataFrame([patient_dic]).to_csv(csv_path, index=False)
        else:
            pd.DataFrame([patient_dic]).to_csv(csv_path, index=False, mode="a", header=False)

    def save_scores(self, scores):
        self.scores  = scores
        np.save(os.path.join(self.save_path, "scores.npy"), scores)

    def get_scores(self):
        self.scores = np.load(os.path.join(self.save_path, "scores.npy"))

    def save_times(self, times):
        self.times  = times
        np.save(os.path.join(self.save_path, "times.npy"), times)

    def get_times(self):
        self.times = np.load(os.path.join(self.save_path, "times.npy"))

    def get_processed_df(self):
        if self.processed_df is None:
            self.processed_df = pd.read_csv(os.path.join(self.save_path, "processed_df.csv"))
        return self.processed_df

    @classmethod
    def load(cls, root_dir, subject_id, hadm_id):
        """
        Load the patient's data from the specified directory.
        """
        # Load patient information
        with open(os.path.join(root_dir, subject_id, hadm_id, "patient_info.json"), "r") as f:
            patient_info = json.load(f)

        patient_info = pd.DataFrame(patient_info)
        patient_info["time_start"] = string_to_timestamp(patient_info["time_start"])

        # Load medical data
        # medical_data_path = os.path.join(load_dir, "medical_data.csv")
        # medical_data = pd.read_csv(medical_data_path) if os.path.exists(medical_data_path) else pd.DataFrame()

        # Return an instance of the Patient class
        return cls(
            df=patient_info,
            root_path=root_dir,
            is_load=True
        )

    def save_notes(self, times, texts):
        self.note_times = times
        self.note_texts = texts
        self.df_notes = pd.DataFrame({
            "times": self.note_times,
            "texts": self.note_texts
        })
        self.df_notes.to_csv(os.path.join(self.save_path, "df_notes.csv"), index=False)

    def get_notes(self):
        df_notes = pd.read_csv(os.path.join(self.save_path, "df_notes.csv"))
        self.note_times = df_notes["times"]
        self.note_texts = df_notes["texts"]


    def get_existing_config(self):
        if os.path.exists(os.path.join(self.save_path, f"config.json")):
            with open(os.path.join(self.save_path, f"config.json"), "r") as f:
                config = json.load(f)
                if "analysis_parameters" not in config:
                    config["analysis_parameters"] = {}
                return config
        return {"analysis_parameters": {}}

    def save_config(self, config):
        with open(os.path.join(self.save_path, f"config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def save_final_df(self, df):
        self.final_df = df
        df.to_csv(os.path.join(self.save_path, "final_df.csv"), index=False)

    def get_final_df(self):
        self.final_df = pd.read_csv(os.path.join(self.save_path, "final_df.csv"))
        return self.final_df

    def __repr__(self):
        return f"Patient(patient_id={self.patient_id}, encounter_id={self.encounter_id}, hadm_id={self.hadm_id})"
import shutil

import numpy as np
import pandas as pd
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import timedelta
import re
from src.utils.patient import Patient
from src.utils.db_utils import run_query

load_dotenv()  # Load environment variables from a .env file

# Main script to analyze patient data from a specified directory
if __name__ == "__main__":
    # Define project directory and load list of patients from CSV file
    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))

    # Define directory for saving all plots from multiple patients
    all_plots_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new_all"

    # Initialize lists to store analysis results
    scores_ratio1 = []
    scores_ratio2 = []
    is_after_admission = []
    is_after_enough_data = []

    # Loop over each patient in the list to process their data
    for index, row in patients_list_df.iterrows():
        # Load patient data using their subject and admission IDs
        patient = Patient.load(project_dir, str(row["subject_id"]), str(row["hadm_id"]))
        print("Starting patient ", patient.subject_id)

        try:
            # Copy plot files for the patient into centralized directories for scores and data plots
            shutil.copy(os.path.join(patient.save_path, "plots", "scores.png"),
                        os.path.join(all_plots_dir, "scores_plots", f"scores_{patient.subject_id}.png"))
            shutil.copy(os.path.join(patient.save_path, "plots", "data.png"),
                        os.path.join(all_plots_dir, "data_plots", f"data_{patient.subject_id}.png"))

            # Retrieve patient times and scores for analysis
            patient.get_times()
            patient.get_scores()

            # Access the existing configuration for analysis parameters
            config = patient.get_existing_config()
            analysis_parameters = config["analysis_parameters"]

            try:
                # Retrieve ARDS onset time if available
                ards_onset_time = config["ards_onset_time"]

                # Determine if ARDS onset is after patient admission
                if ards_onset_time is not None and ards_onset_time > 0:
                    is_after_admission.append(1)
                elif ards_onset_time is not None:
                    is_after_admission.append(0)

                # Check if ARDS onset time is after sufficient data has been collected
                if ards_onset_time is not None and ards_onset_time > patient.times[0]:
                    is_after_enough_data.append(1)
                elif ards_onset_time is not None:
                    is_after_enough_data.append(0)

                # Set a time shift for measuring scores around the ARDS onset time
                time_shift = 10

                # Analyze scores around the ARDS onset time, if it exists
                if ards_onset_time != 0:
                    try:
                        # Locate indices before, at, and after the ARDS onset time
                        ards_onset_row = np.where(patient.times > ards_onset_time)[0][0]
                        ards_hbefore = np.where(patient.times > ards_onset_time - time_shift)[0][0]
                        ards_hafter = np.where(patient.times > ards_onset_time + time_shift)[0][0]

                        # Calculate log-transformed scores around onset if all scores are negative
                        if patient.scores[ards_onset_row] < 0 and patient.scores[ards_hbefore] < 0 and patient.scores[
                            ards_hafter] < 0:
                            score_onset = -np.log(-patient.scores[ards_onset_row])
                            score_hbefore = -np.log(-patient.scores[ards_hbefore])
                            score_hafter = -np.log(-patient.scores[ards_hafter])

                            # Calculate ratios of scores and append to the lists
                            scores_ratio1.append(score_onset / score_hbefore)
                            scores_ratio2.append(score_hafter / score_onset)

                    except Exception as e:
                        # Skip any exceptions encountered in calculating ratios
                        pass

            except KeyError:
                # Handle missing ARDS onset information in configuration
                print(config)
        except FileNotFoundError:
            # Skip if necessary files are not found for the patient
            pass

    # Print statistical summaries of the collected score ratios and ARDS timing data
    print(f"First ratio mean :", np.mean(scores_ratio1))
    print(f"Second ratio mean:", np.mean(scores_ratio2))
    print(f"Ratio of onset after admission :", np.mean(is_after_admission))
    print(f"Ratio of onset after enough data :", np.mean(is_after_enough_data))

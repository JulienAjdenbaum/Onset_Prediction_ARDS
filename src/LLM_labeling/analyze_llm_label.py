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
load_dotenv()



if __name__ == "__main__":
    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))
    all_plots_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new_all"
    # if not os.path.exists(os.path.join(all_plots_dir, "scores")):
    scores_ratio1 = []
    scores_ratio2 = []
    is_after_admission = []
    is_after_enough_data = []
    for index, row in patients_list_df.iterrows():
        patient = Patient.load(project_dir, str(row["subject_id"]), str(row["hadm_id"]))
        print("Starting patient ", patient.subject_id)
        try:

            shutil.copy(os.path.join(patient.save_path, "plots", "scores.png"), os.path.join(all_plots_dir, "scores_plots", f"scores_{patient.subject_id}.png"))
            shutil.copy(os.path.join(patient.save_path, "plots", "data.png"), os.path.join(all_plots_dir, "data_plots", f"data_{patient.subject_id}.png"))
            patient.get_times()
            patient.get_scores()
            config = patient.get_existing_config()
            analysis_parameters = config["analysis_parameters"]
            try:
                ards_onset_time = config["ards_onset_time"]
                if ards_onset_time is not None and ards_onset_time > 0:
                    is_after_admission.append(1)
                elif ards_onset_time is not None:
                    is_after_admission.append(0)
                if ards_onset_time is not None and ards_onset_time > patient.times[0]:
                    is_after_enough_data.append(1)
                elif ards_onset_time is not None:
                    is_after_enough_data.append(0)
                # print(ards_onset_time)
                time_shift = 10
                if ards_onset_time != 0:
                    try:
                        ards_onset_row = np.where(patient.times > ards_onset_time)[0][0]
                        ards_hbefore = np.where(patient.times > ards_onset_time-time_shift)[0][0]
                        ards_hafter = np.where(patient.times > ards_onset_time+time_shift)[0][0]
                        if patient.scores[ards_onset_row] < 0 and patient.scores[ards_hbefore] < 0 and patient.scores[ards_hafter] < 0:
                            score_onset = -np.log(-patient.scores[ards_onset_row])
                            score_hbefore = -np.log(-patient.scores[ards_hbefore])
                            score_hafter = -np.log(-patient.scores[ards_hafter])
                            # if not score_onset.isnan() and not score_hbefore.isnan() and not score_hafter.isnan():
                            # print(score_onset, score_hbefore, score_hafter)
                            scores_ratio1.append(score_onset/score_hbefore)
                            scores_ratio2.append(score_hafter/score_onset)
                            # print(score_onset/score_hbefore)
                            # print(score_hafter/score_onset)
                    except Exception as e:
                        # print(e)
                        pass
                # if patient.times[ards_onset_time] > 0 and ards_onset_time:
                #     print(patient.scores[ards_onset_time])

            except KeyError:
                # print(analysis_parameters)
                print(config)
        except FileNotFoundError:
            # print(patient.subject_id)
            pass
    print(f"First ratio mean :", np.mean(scores_ratio1))
    print(f"Second ration mean:", np.mean(scores_ratio2))
    print(f"Ratio of onset after admission :", np.mean(is_after_admission))
    print(f"Ration of onset after enough data :", np.mean(is_after_enough_data))
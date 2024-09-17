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
    for index, row in patients_list_df.iterrows():
        patient = Patient.load(project_dir, str(row["subject_id"]), str(row["hadm_id"]))
        shutil.copy(os.path.join(patient.save_path, "plots", "scores.png"), os.path.join(all_plots_dir, "scores_plots", f"scores_{patient.subject_id}.png"))
        shutil.copy(os.path.join(patient.save_path, "plots", "data.png"), os.path.join(all_plots_dir, "data_plots", f"data_{patient.subject_id}.png"))

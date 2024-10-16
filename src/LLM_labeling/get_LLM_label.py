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

@dataclass
class ARDSOnsetResponse:
    onset_timestamp: timedelta
    confidence: str
    # summary: str
    evidence: List[str] = field(default_factory=list)
    additional_notes: List[str] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, response_text: str) -> 'ARDSOnsetResponse':
        # Parse the API response text and create an instance of ARDSOnsetResponse
        sections = re.split(r'#{3,4}', response_text)

        onset_timestamp = sections[1].split(':')[1].strip()
        evidence = cls._parse_list(sections[2])
        confidence = sections[3].split(':')[1].strip()
        additional_notes = cls._parse_list(sections[4])
        # summary = sections[5].replace('### Summary', '').strip()

        return cls(
            onset_timestamp=onset_timestamp,
            confidence=confidence,
            # summary=summary,
            evidence=evidence,
            additional_notes=additional_notes
        )

    @staticmethod
    def _parse_list(section: str) -> List[str]:
        return [item.strip('- ').strip() for item in section.split('\n') if item.strip()]

    def to_dict(self) -> dict:
        return {
            'onset_timestamp': str(self.onset_timestamp),
            'confidence': self.confidence,
            # 'summary': self.summary,
            'evidence': self.evidence,
            'additional_notes': self.additional_notes
        }

    def __str__(self) -> str:
        return f"""ARDS Onset Analysis:
Onset Timestamp: {self.onset_timestamp}
Confidence: {self.confidence}
Summary: {self.summary}
Evidence: {', '.join(self.evidence)}
Additional Notes: {', '.join(self.additional_notes)}"""



class Sentence(BaseModel):
    privacy: str
    sentence: str

class PrivacySentences(BaseModel):
    sentences: list[Sentence]


caregivers = {}

prompt = """
You are an AI assistant specialized in analyzing complex medical notes. You will be given a dictionary where keys are timestamps and values are detailed doctor's notes. These notes may be very long and contain various special characters, medical abbreviations, and technical terms. Your task is to identify the earliest indication of Acute Respiratory Distress Syndrome (ARDS) onset.

When analyzing each note:
1. Process the entire note, regardless of length.
2. Ignore irrelevant special characters, focusing on the medical content.
3. Interpret common medical abbreviations and technical terms related to ARDS.
4. Look for key indicators of ARDS, including but not limited to:
   a) Direct mentions of "ARDS" or "Acute Respiratory Distress Syndrome"
   b) Clinical signs: severe dyspnea, rapid/labored breathing, hypoxemia
   c) Diagnostic findings: bilateral infiltrates on imaging, PaO2/FiO2 ratio ≤ 300 mmHg
   d) ARDS risk factors: sepsis, pneumonia, severe trauma, aspiration
   e) ARDS-specific treatments: mechanical ventilation with high PEEP, prone positioning

Analysis process:
1. Examine each note chronologically.
2. For each note, determine if there's evidence of ARDS onset.
3. If found, record the timestamp and the specific indicators.
4. Continue through all notes to ensure you've identified the earliest onset.

Output format:
#### ARDS Onset Timestamp: [Insert earliest timestamp indicating ARDS]
#### Evidence: [Brief summary of the key information indicating ARDS onset]
#### Confidence: [High/Medium/Low, based on the clarity of the evidence]
#### Additional Notes: [Any relevant observations or uncertainties]

If no clear indication of ARDS is found, state: "No definitive ARDS onset could be determined from the provided notes."

Remember:
- Focus on medical content, not formatting issues.
- Be thorough but concise in your analysis.
- If uncertain, err on the side of caution and explain your reasoning.

Input Format:
The medical notes will be provided in the following Python dictionary format:

medical_notes = {
    \"timestamp1\": \"\"\"
    [Full text of doctor's note 1]
    \"\"\",
    \"timestamp2\": \"\"\"
    [Full text of doctor's note 2]
    \"\"\",
    # ... more entries ...
}

Please analyze these notes to determine the onset of ARDS as per the instructions provided earlier.
"""

def get_patient_idm_date(hadm_id):
    query = """
            SELECT admittime
            FROM ADMISSIONS
            WHERE hadm_id = %(hadm_id)s
            LIMIT 1
        """
    df = run_query(query, {"hadm_id": hadm_id})
    # print(df)
    return df["admittime"].iloc[0]


def load_patient_data(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data


def get_patients_notes(hadm_id, start_time=None):
    query = """
        SELECT charttime, text
        FROM NOTEEVENTS
        WHERE hadm_id = %(hadm_id)s
    """
    df = run_query(query, {"hadm_id": hadm_id})
    if start_time is None:
        df["charttime"] = df["charttime"] - get_patient_idm_date(hadm_id)
    else:
        df["charttime"] = df["charttime"] - start_time.values[0]
    # print(df)
    return list(map(str, df["charttime"])), df["text"]

def get_LLM_result(patient):
    if os.path.exists(os.path.join(patient.save_path, "OpenAI.json")):
        print("OpenAI JSON found")
        with open(os.path.join(patient.save_path, "OpenAI.json"), 'r') as f:
            ards_response = json.load(f)
    else:
        print("No OpenAI JSON found")
        try:
            ards_response = send_request(patient).to_dict()
            with open(os.path.join(patient.save_path, "OpenAI.json"), "w") as f:
                json.dump(ards_response, f, indent=4)
        except Exception as e:
            print(f"Rate limit error : {e}")
            return {"onset_timestamp": "None", "confidence": "None",}
    return ards_response


def send_request(patient):
    times, texts = get_patients_notes(patient.hadm_id, patient.time_start)
    patient.save_notes(times, texts)
    notes_dict = dict(zip(times, texts))

    client = OpenAI()

    chunk_text = "medical_notes = {"
    for time in notes_dict:
        chunk_text += f"\"{time}\": \"\"\"\n"
        chunk_text += notes_dict[time]
        chunk_text += f"\"\"\",\n"
    chunk_text += "}\n"

    # print(chunk_text)
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system",
                   "content": prompt
                   },
                  {"role": "user",
                   "content": chunk_text}],
        # response_format=PrivacySentences,
        temperature=0.0,
        # max_tokens=4096
    )
    results = response.choices[0].message
    ards_response = ARDSOnsetResponse.from_api_response(results.content)

    # Now you can access the structured data
    return ards_response


if __name__ == "__main__":
    project_dir = "/home/julien/Documents/stage/data/MIMIC/cohorts_new"
    patients_list_df = pd.read_csv(os.path.join(project_dir, "patients.csv"))
    for index, row in patients_list_df.iterrows():
        patient = Patient.load(project_dir, str(row["subject_id"]), str(row["hadm_id"]))
        run_PCA = True
        get_LLM_result(patient)
        break

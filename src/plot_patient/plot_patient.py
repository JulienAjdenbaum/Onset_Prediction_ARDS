import json
import os
import subprocess
import matplotlib.pyplot as plt
import shutil
import numpy as np
import warnings
import re
from src.plot_patient.get_patients_notes import get_patients_notes
from src.LLM_labeling.get_LLM_label import get_LLM_result

from dotenv import load_dotenv

# Function to load patient data
def load_patient_data(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data


# Function to create a LaTeX config file
def create_latex_config(patient_data, ehr_plot, algo_plot, template_dir, patient_dir):
    if os.path.exists(os.path.join(patient_dir, "latex")):
        shutil.rmtree(os.path.join(patient_dir, "latex"))
    os.makedirs(os.path.join(patient_dir, "latex"))

    os.makedirs(os.path.join(patient_dir, "latex", "plots"))

    shutil.copy(os.path.join(patient_dir, "plots", "data.png"), os.path.join(patient_dir, "latex", "plots", "data.png"))
    shutil.copy(os.path.join(patient_dir, "plots", "scores.png"), os.path.join(patient_dir, "latex", "plots", "scores.png"))

    patient_dir_latex = os.path.join(patient_dir, "latex")
    shutil.copy(os.path.join(template_dir, "main.tex"), os.path.join(patient_dir_latex, "main.tex"))

    os.makedirs(os.path.join(patient_dir_latex, "sections"))

    with open(os.path.join(template_dir, "sections", "intro.tex"), 'r') as file:
        template = file.read()
    filled_template = template % {
        'patient_id': patient_data['patient_id'],
        'encounter_id': patient_data['encounter_id'],
        'stay_duration': np.round(patient_data['stay_duration'], 1),
        'number_of_measurements': patient_data['number_of_measurements'],}
    with open(os.path.join(patient_dir_latex, "sections", "intro.tex"), 'w') as f:
        f.write(filled_template)

    with open(os.path.join(template_dir, "sections", "features.tex"), 'r') as file:
        template = file.read()
    features_txt = ""
    for i, feature in enumerate(patient_data['fit_keys']):
        if i%3 == 2:
            features_txt += f"\\textbf{{{feature}}} \\\\ \n"
        else:
            features_txt += f"\\textbf{{{feature}}} & "
    filled_template = template % {
        "items": features_txt}
    with open(os.path.join(patient_dir_latex, "sections", "features.tex"), 'w') as f:
        f.write(filled_template)

    with open(os.path.join(template_dir, "sections", "analysis_parameters.tex"), 'r') as file:
        template = file.read()
    items_txt = ""
    for i, feature in enumerate(patient_data['analysis_parameters']):
        name = feature.replace("_", " ")
        value = patient_data["analysis_parameters"][feature]
        items_txt += f"\\item \\textbf{{{name}}}: {value} \n"
    filled_template = template % {
        "items":items_txt}

    # filled_template = template % {
    #     "first_valid_index":patient_data['first_valid_index'],
    #     "df_interest_start":patient_data['df_interest_start'] + patient_data['first_valid_index'],
    #     "window_length":patient_data['window_length'],
    #     "max_nan_percentage":patient_data['max_nan_percentage'],
    #     "fit_ends":patient_data['window_length_fit']+patient_data['df_interest_start'] + patient_data['first_valid_index']}
    with open(os.path.join(patient_dir_latex, "sections", "analysis_parameters.tex"), 'w') as f:
        f.write(filled_template)

    with open(os.path.join(template_dir, "sections", "plots.tex"), 'r') as file:
        template = file.read()
    filled_template = template % {
        "plot1_path":ehr_plot,
        "plot2_path":algo_plot}
    with open(os.path.join(patient_dir_latex, "sections", "plots.tex"), 'w') as f:
        f.write(filled_template)

    with open(os.path.join(template_dir, "sections", "diagnoses.tex"), 'r') as file:
        template = file.read()
    items_txt = ""
    for i, feature in enumerate(patient_data['diagnoses']):
        items_txt += f"\\item \\textbf{{{feature}}} \n"
    filled_template = template % {
        "items":items_txt}
    with open(os.path.join(patient_dir_latex, "sections", "diagnoses.tex"), 'w') as f:
        f.write(filled_template)

    with open(os.path.join(template_dir, "sections", "notes.tex"), 'r') as file:
        template = file.read()
    hadm_id = patient_data["hadm_id"]
    categories, descriptions, times, clinicians, notes = get_patients_notes(hadm_id)
    filled_template = ""
    for category, description, time, clinician, note in zip(categories, descriptions, times, clinicians, notes):
        filled_template += template % {
            "category":category,
            "description":description,
            "time":time,
            "clinician":clinician,
            "content":note,
            }
    with open(os.path.join(patient_dir_latex, "sections", "notes.tex"), 'w') as f:
        f.write(filled_template)
    print(patient_dir_latex)
    llm_result = get_LLM_result(patient_dir, patient_data["hadm_id"])

    with open(os.path.join(template_dir, "sections", "lllm_results.tex"), 'r') as file:
        template = file.read()

    print(llm_result['onset_timestamp'])
    # print(llm_result.evidence)
    print(llm_result['confidence'])
    # print(llm_result.additional_notes)

    content = "Evidence: \n"
    content += ' '.join(llm_result['evidence'][1:])
    content += "\n"
    content += "Additional Notes: \n\n"
    content += ' '.join(llm_result['additional_notes'][1:])

    print(content)
    filled_template = template % {
        "time":llm_result['onset_timestamp'],
        "confidence":llm_result['confidence'],
        "content":content}
    with open(os.path.join(patient_dir_latex, "sections", "lllm_results.tex"), 'w') as f:
        f.write(filled_template)




# Function to compile LaTeX document
def compile_latex(latex_dir, latex_template):
    os.chdir(latex_dir)
    subprocess.run(['pwd'], stdout = subprocess.DEVNULL)
    subprocess.run(['pdflatex', '-interaction=nonstopmode', latex_template, '-jobname', 'report.pdf'])
    subprocess.run(['xdg-open', 'main.pdf'])
    # while 1: pass
    # subprocess.run(['pdflatex', '-interaction=nonstopmode', latex_template, '-output-directory', output_dir, '-jobname', 'report.pdf'])


if __name__ == "__main__":
    load_dotenv()
    home = "/home/julien/Documents/stage"
    # print(home.stdout)
    save_dir = "data/MIMIC/cohorts_new"
    save_pdf_all_dir = "data/MIMIC/cohorts_new_all/pdf"
    save_ehr_all_dir = "data/MIMIC/cohorts_new_all/data_plots"
    save_scores_all_dir = "data/MIMIC/cohorts_new_all/scores_plots"
    for patient_unique_dir in os.listdir(save_dir):
        for patient_dir_last in os.listdir(os.path.join(save_dir, patient_unique_dir)):
            patient_dir = os.path.join(save_dir, patient_unique_dir, patient_dir_last)

            patient_id, encounter_id = [int(num) for num in re.findall(r'\d+', patient_dir)]
            print(f"Generating plot for patient {patient_id}")
            latex_dir = "src/plot_patient/latex"
            ehr_plot_path = os.path.join("plots", "data.png")
            algo_plot_path = os.path.join("plots", "scores.png")

            try:
                patient_data = load_patient_data(os.path.join(patient_dir, "saves", "config.json"))
            except FileNotFoundError:
                warnings.warn(f"Skipping bc file not found : {os.path.join(patient_dir, 'saves', 'config.json')}")
                continue
            # try:
            create_latex_config(patient_data, ehr_plot_path, algo_plot_path, latex_dir, patient_dir)
            # except FileNotFoundError:
            #     warnings.warn(f"Skipping bc file not found : {os.path.join(patient_dir, 'saves', 'config.json')}")
            #     continue
            compile_latex(os.path.join(patient_dir, "latex"), "main.tex")
            os.chdir(home)
            shutil.copy(os.path.join(patient_dir, "latex", "main.pdf"),
                        os.path.join(save_pdf_all_dir, f"{patient_id}_{encounter_id}.pdf"))
            shutil.copy(os.path.join(patient_dir, ehr_plot_path),
                        os.path.join(save_ehr_all_dir, f"{patient_id}_{encounter_id}.png"))
            shutil.copy(os.path.join(patient_dir, algo_plot_path),
                        os.path.join(save_scores_all_dir, f"{patient_id}_{encounter_id}.png"))
        #     break
        # break
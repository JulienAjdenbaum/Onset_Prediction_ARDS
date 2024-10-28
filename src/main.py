from src.ARDS_mimic import get_cohort, reformat_data, preprocessing
from src.LLM_labeling import get_LLM_label
from src.models import run_models


if __name__ == '__main__':
    plot = True

    critical_measurements = ["Heart Rate", "SpO2", "Respiratory Rate", "Arterial BP [Systolic]",
                             "Arterial BP [Diastolic]",
                             "Temperature F", 'PEEP Set', 'FiO2 Set', 'Arterial PaO2']

    save = True
    dataset = "carevue"
    project_dir = "data/MIMIC/final"

    get_cohort.main(critical_measurements, project_dir, save, plot)
    reformat_data.main(project_dir)
    preprocessing.main(project_dir, critical_measurements)
    get_LLM_label.main(project_dir)
    run_models.main(project_dir)

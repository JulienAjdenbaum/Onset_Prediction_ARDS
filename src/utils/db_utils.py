import json
from sqlalchemy import create_engine
import pandas as pd


# Load database configuration from JSON file
def load_db_config(file_path='db_config.json'):
    with open(file_path, 'r') as config_file:
        return json.load(config_file)


# Create the engine once and reuse it
def create_db_engine(db_config):
    return create_engine(
        f"postgresql://{db_config['db_user']}:{db_config['db_password']}@{db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}"
    )


# Example usage
db_config = load_db_config()
engine = create_db_engine(db_config)  # Create engine once


def run_query(query: str, params: dict, schema_name: str = db_config['schema_name']) -> pd.DataFrame:
    """
    Execute an SQL query using a single engine connection and return the result as a DataFrame.

    Parameters:
    query (str): The SQL query to execute.
    params (dict): A dictionary of parameters to bind to the query.
    schema_name (str): The schema to set for the search path.

    Returns:
    pd.DataFrame: The result of the query as a pandas DataFrame.
    """
    with engine.connect() as connection:
        try:
            df = pd.read_sql_query(query, connection, params=params)
        except Exception as e:
            raise
    return df

if __name__ == '__main__':
    # Define query and parameters
    query = """
        SELECT long_title
        FROM DIAGNOSES_ICD
        JOIN D_ICD_DIAGNOSES ON DIAGNOSES_ICD.icd9_code = D_ICD_DIAGNOSES.icd9_code
        WHERE subject_id = %(subject_id)s AND hadm_id = %(hadm_id)s
        """

    params = {'subject_id': 117, 'hadm_id': 164853}

    # Call the function
    df = run_query(query, params)
    print(df.head())
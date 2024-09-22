import json
from sqlalchemy import create_engine, text
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

def create_cohort_table(icd9_codes, batch_size=100000):
    icd9_str = ', '.join(f"'{code}'" for code in icd9_codes)

    with engine.connect() as conn:
        query = f"""
        SELECT COUNT(*)
        FROM CHARTEVENTS ce
        JOIN DIAGNOSES_ICD di
          ON ce.hadm_id = di.hadm_id
        WHERE di.icd9_code IN ({icd9_str})
        """

        result = conn.execute(text(query))
        print(f"There is {result.scalar()} rows to run.")

    offset = 0
    while True:
        # Query the DIAGNOSES_ICD and CHARTEVENTS for a batch
        with engine.connect() as conn:
            query = f"""
            INSERT INTO CHARTEVENTS_COHORTS
            SELECT ce.subject_id, ce.hadm_id, ce.charttime, ce.itemid, ce.value
            FROM CHARTEVENTS ce
            JOIN DIAGNOSES_ICD di
              ON ce.hadm_id = di.hadm_id
            WHERE di.icd9_code IN ({icd9_str})
            LIMIT {batch_size} OFFSET {offset};
            """
            # Execute the query
            result = conn.execute(text(query))

            # If no rows were returned, stop the loop
            if result.rowcount == 0:
                break

            # Increase the offset for the next batch
            offset += batch_size
            print(f"Processed batch starting from offset {offset}")
            conn.commit()

if __name__ == '__main__':
    codes = ('51881', '51882', '51884', '5185', '51851')
    create_cohort_table(codes)
    # Define query and parameters
    # query = """
    #     SELECT long_title
    #     FROM DIAGNOSES_ICD
    #     JOIN D_ICD_DIAGNOSES ON DIAGNOSES_ICD.icd9_code = D_ICD_DIAGNOSES.icd9_code
    #     WHERE subject_id = %(subject_id)s AND hadm_id = %(hadm_id)s
    #     """

    # params = {'subject_id': 117, 'hadm_id': 164853}
    #
    # # Call the function
    # df = run_query(query, params)
    # print(df.head())


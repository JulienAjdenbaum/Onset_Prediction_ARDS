import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2

# Setup the database connection
db_url = "postgresql://username:password@localhost:5432/your_database"
engine = create_engine(db_url)


# Function to insert data in batches
def insert_data_in_batches(batch_size):
    # Start with a SQLAlchemy connection
    with engine.connect() as conn:
        offset = 0
        while True:
            # Query the DIAGNOSES_ICD and CHARTEVENTS for a batch
            query = f"""
            INSERT INTO CHARTEVENTS_COHORTS
            SELECT ce.subject_id, ce.hadm_id, ce.charttime, ce.itemid, ce.value
            FROM CHARTEVENTS ce
            JOIN DIAGNOSES_ICD di
              ON ce.hadm_id = di.hadm_id
            WHERE di.icd9_code IN ('51881', '51882', '51884', '5185', '51851')
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


# Set the batch size (e.g., 10,000 rows per batch)
batch_size = 10000

# Run the function to insert data in batches
insert_data_in_batches(batch_size)

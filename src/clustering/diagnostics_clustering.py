from src.utils.db_utils import run_query
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


def get_diagnostic_df(cohort):
    cohort_tuple = tuple(cohort)

    query = """
            SELECT hadm_id, seq_num, icd9_code
            FROM diagnoses_icd
            WHERE hadm_id IN (
                SELECT hadm_id
                FROM diagnoses_icd
                WHERE icd9_code IN %(cohort)s)"""

    # Correctly pass the tuple to the query
    diagnostics_list = run_query(query, {"cohort": cohort_tuple})
    print(diagnostics_list.describe())
    return pd.pivot_table(diagnostics_list, values='seq_num', index='hadm_id',
                          columns='icd9_code', aggfunc='sum', fill_value=0)


def analyze_diagnostics(cohort):
    df = get_diagnostic_df(cohort)
    print(len(df))
    print(len(np.unique(df.index)))
    # print(df.columns[])
    for hadm_id, row in df.iterrows():
        row_ARDS = row[list(cohort)]
        print(
            f"Patient {hadm_id} has {(row_ARDS != 0).sum()} ARDS diagnostics, the most important one being {row_ARDS[row_ARDS != 0].idxmin()} with seq_num {row_ARDS[row_ARDS != 0].min()}")


def cluster_patients(cohort, n_clusters=10):
    # Ensure the cohort is passed as a tuple for proper SQL handling
    df_pivot = get_diagnostic_df(cohort)
    # Apply TF-IDF to the pivoted DataFrame
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(df_pivot)

    # Perform clustering using KMeans (or other clustering algorithms)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df_pivot['cluster_tfidf'] = kmeans.fit_predict(tfidf_matrix.toarray())

    # Use PCA for dimensionality reduction and visualize the new clustering
    pca_tfidf = PCA(n_components=2)
    reduced_data_tfidf = pca_tfidf.fit_transform(tfidf_matrix.toarray())

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data_tfidf[:, 0], reduced_data_tfidf[:, 1], c=df_pivot['cluster_tfidf'], cmap='viridis', s=100)
    plt.title("KMeans Clustering of Hospital Admissions with TF-IDF Encoding")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

    # print(diagnostics_list.head())


if __name__ == '__main__':
    ARDS_list = ('51881', '51882', '51884', '5185')  # Ensure codes are strings
    cluster_patients(ARDS_list)
    # analyze_diagnostics(ARDS_list)

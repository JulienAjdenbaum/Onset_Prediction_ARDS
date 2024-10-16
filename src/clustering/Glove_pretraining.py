import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import numpy as np


# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# Function to Pre-train Autoencoder
def pretrain_ae(autoencoder, data, epochs=50, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstruction, _ = autoencoder(data)
        loss = criterion(reconstruction, data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


# Function to Apply UMAP
def apply_umap(latent_vectors, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(latent_vectors)
    return embedding


# Function to Perform Clustering
def perform_clustering(latent_vectors, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    silhouette_avg = silhouette_score(latent_vectors, cluster_labels)
    print(f'Silhouette Score: {silhouette_avg}')
    return cluster_labels


# Example Usage
input_dim = 3995  # Adjust based on your input dimensions
latent_dim = 64  # Dimensionality of latent space

# Assume `icd_embeddings` is your GloVe-pretrained ICD embeddings, loaded as a tensor
autoencoder = Autoencoder(input_dim, latent_dim)

# Pretrain Autoencoder with ICD embeddings
pretrain_ae(autoencoder, icd_embeddings, epochs=100)

# Extract latent vectors
_, latent_vectors = autoencoder(icd_embeddings)

# Apply UMAP for dimensionality reduction
umap_embeddings = apply_umap(latent_vectors.detach().numpy())

# Perform K-means clustering
cluster_labels = perform_clustering(umap_embeddings, n_clusters=3)

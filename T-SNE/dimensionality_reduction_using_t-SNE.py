# 1. Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

# 2. Load dataset (Iris as an example)
# X = features, y = class labels
X, y = load_iris(return_X_y=True)

# 3. Standardize the data
# IMPORTANT: t-SNE is very sensitive to feature scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Initialize and run t-SNE
# n_components = 2 for 2D visualization
# perplexity controls local neighborhood size
# metric defines distance calculation
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    metric="mahalanobis",   # try "cosine" for comparison
    random_state=42
)

# Fit t-SNE and transform data
X_tsne = tsne.fit_transform(X_scaled)

# 5. Compute evaluation metrics

# 5.1 Trustworthiness (best metric for t-SNE)
# Measures how well local neighborhoods are preserved
trust = trustworthiness(X_scaled, X_tsne, n_neighbors=5)
print(f"Trustworthiness score: {trust:.4f}")

# 5.2 Silhouette score (optional, label-dependent)
# Measures how well classes are separated in embedding
sil_score = silhouette_score(X_tsne, y)
print(f"Silhouette score: {sil_score:.4f}")

# 6. Visualization of t-SNE results
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_tsne[:, 0],          # t-SNE dimension 1
    X_tsne[:, 1],          # t-SNE dimension 2
    c=y,                   # color by class label
    cmap="viridis",
    s=60,
    alpha=0.8
)

# Add color bar and labels
plt.colorbar(scatter, label="Class Label")
plt.title("t-SNE Visualization (Iris Dataset)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)

# Display plot
plt.show()

import pandas as pd
url = "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/tests/io/data/csv/tips.csv" # Corrected URL to an existing raw CSV file
data = pd.read_csv(url)
print(data.head())
data.info()
data.describe()

import pandas as pd
url = "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/tests/io/data/csv/iris.csv"
data = pd.read_csv(url)
print(data.head())
data.info()
data.describe()

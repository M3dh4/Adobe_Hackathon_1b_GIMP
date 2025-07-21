import numpy as np
from sklearn.cluster import KMeans

def cluster_font_attributes(X, n_clusters=3):
    """
    Cluster font attributes (size, bold, left) to assign heading levels.
    Returns cluster labels: 0=body, 1=H3, 2=H2, 3=H1 (sorted by font size)
    """
    if len(X) < n_clusters:
        return np.zeros(len(X), dtype=int)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    # Sort clusters by mean font size (largest = H1)
    means = [X[labels == i, 0].mean() for i in range(n_clusters)]
    order = np.argsort(means)[::-1]
    label_map = {old: new for new, old in enumerate(order)}
    sorted_labels = np.array([label_map[l] for l in labels])
    return sorted_labels 
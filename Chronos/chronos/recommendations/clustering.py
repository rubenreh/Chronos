"""
User clustering for matching similar productivity profiles in Chronos.

This module groups users into clusters based on their behavioural feature
vectors (extracted by chronos.features.extractor). Clustering enables:

  • "Users like you" recommendations — find peers with similar patterns.
  • Cohort analytics — compare cluster-level statistics on the dashboard.
  • Personalisation — tailor coaching advice to a cluster archetype.

Two algorithms are supported:
  KMeans  — partition-based; requires a pre-set number of clusters (k).
  DBSCAN  — density-based; discovers clusters automatically but needs
             eps (neighbourhood radius) and min_samples.

The module follows an sklearn-style fit / predict / score API.
"""

import numpy as np                                   # Array operations for feature matrices
import pandas as pd                                  # Series input type for time-series data
from typing import List, Dict, Optional              # Type annotations
from sklearn.cluster import KMeans, DBSCAN           # Two clustering algorithms
from sklearn.preprocessing import StandardScaler     # Feature normalisation (zero-mean, unit-var)
from sklearn.metrics import silhouette_score          # Cluster quality metric in [-1, 1]

from chronos.features.extractor import extract_behavioral_features  # Feature extraction helper


class UserClustering:
    """Cluster users by their productivity behaviour patterns.

    Wraps KMeans or DBSCAN with automatic feature extraction and optional
    StandardScaler normalisation.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        method: str = 'kmeans',
        normalize: bool = True
    ):
        """Initialise clustering configuration.

        Args:
            n_clusters: Target number of clusters (used by KMeans only; DBSCAN
                        discovers its own cluster count).
            method: 'kmeans' for centroid-based or 'dbscan' for density-based.
            normalize: If True, z-score normalise features before clustering to
                       prevent features with large magnitudes from dominating.
        """
        self.n_clusters = n_clusters          # Stored for KMeans initialisation
        self.method = method                  # Clustering algorithm identifier
        self.normalize = normalize            # Whether to apply StandardScaler
        self.scaler = StandardScaler() if normalize else None  # Scaler instance (or None)
        self.clusterer = None                 # Will hold the fitted KMeans or DBSCAN object
        self.feature_names_ = None            # List of feature names after first fit

    def fit(self, user_series: List[pd.Series]):
        """Fit the clustering model on a list of user time-series.

        Extracts behavioural features from every user's series, optionally
        normalises them, and fits the chosen clustering algorithm.

        Args:
            user_series: List of pandas Series, one per user.
        """
        # Extract behavioural features for each user
        features_list = []
        for series in user_series:
            features = extract_behavioral_features(series)       # Dict of feature_name → value
            features_list.append(list(features.values()))        # Keep values only, as a list

        if not features_list:
            raise ValueError("No features extracted from user series")

        # Record the feature names from the first user (order is deterministic)
        self.feature_names_ = list(extract_behavioral_features(user_series[0]).keys())
        X = np.array(features_list)  # Convert to 2-D numpy array (n_users, n_features)

        # Normalise feature matrix so each feature has mean=0, std=1
        if self.normalize:
            X = self.scaler.fit_transform(X)

        # Instantiate and fit the selected clustering algorithm
        if self.method == 'kmeans':
            # n_init=10 runs KMeans 10 times with different centroids and picks the best
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.method == 'dbscan':
            # eps=0.5 defines the neighbourhood radius; min_samples=2 is the minimum cluster size
            self.clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        self.clusterer.fit(X)  # Learn cluster assignments from the feature matrix

    def predict(self, series: pd.Series) -> int:
        """Predict which cluster a single user belongs to.

        Args:
            series: The user's productivity time-series.

        Returns:
            Integer cluster label.
        """
        if self.clusterer is None:
            raise ValueError("Clustering model not fitted. Call fit() first.")

        features = extract_behavioral_features(series)      # Extract features for this user
        X = np.array([list(features.values())])              # Shape (1, n_features) for sklearn

        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)  # Apply the same normalisation learned during fit()

        return int(self.clusterer.predict(X)[0])  # Return the predicted cluster label as int

    def get_cluster_centers(self) -> np.ndarray:
        """Return the centroid of each cluster (KMeans only).

        Returns:
            2-D array of shape (n_clusters, n_features).

        Raises:
            ValueError: If the model is not a fitted KMeans.
        """
        if self.method != 'kmeans' or self.clusterer is None:
            raise ValueError("Cluster centers only available for fitted KMeans")
        return self.clusterer.cluster_centers_  # Centroid array from sklearn

    def get_silhouette_score(self, user_series: List[pd.Series]) -> float:
        """Compute the silhouette score to evaluate cluster quality.

        The silhouette score ranges from -1 (poor) to +1 (well-separated clusters).
        A score near 0 means overlapping clusters.

        Args:
            user_series: The same list of series used during fit().

        Returns:
            Float silhouette score, or -1.0 if fewer than 2 clusters exist.
        """
        if self.clusterer is None:
            raise ValueError("Clustering model not fitted")

        # Re-extract features (must match the order used during fit)
        features_list = []
        for series in user_series:
            features = extract_behavioral_features(series)
            features_list.append(list(features.values()))

        X = np.array(features_list)
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)  # Apply same normalisation as fit()

        labels = self.clusterer.labels_  # Cluster assignments from fit()
        # Silhouette score requires at least 2 distinct clusters
        if len(set(labels)) < 2:
            return -1.0  # Not enough clusters for a meaningful silhouette score

        return float(silhouette_score(X, labels))  # Compute and return

    def find_similar_users(
        self,
        query_series: pd.Series,
        all_user_series: List[pd.Series],
        top_k: int = 5
    ) -> List[int]:
        """Find users whose productivity profile is most similar to the query user.

        Strategy:
          1. Predict the query user's cluster.
          2. Return all users in the same cluster.
          3. If there aren't enough same-cluster peers, fall back to Euclidean
             distance in feature space.

        Args:
            query_series: The query user's time-series.
            all_user_series: List of all users' time-series.
            top_k: Number of similar users to return.

        Returns:
            List of integer indices into `all_user_series`.
        """
        query_cluster = self.predict(query_series)  # Which cluster the query user belongs to

        # Collect indices of all users in the same cluster
        similar_indices = []
        for i, series in enumerate(all_user_series):
            cluster = self.predict(series)          # Predict cluster for each user
            if cluster == query_cluster:
                similar_indices.append(i)

        # If the same-cluster set is too small, fall back to feature-space distance
        if len(similar_indices) < top_k:
            query_features = extract_behavioral_features(query_series)
            query_vec = np.array(list(query_features.values()))  # Query feature vector

            similarities = []
            for i, series in enumerate(all_user_series):
                features = extract_behavioral_features(series)
                vec = np.array(list(features.values()))
                # Inverse Euclidean distance as a similarity measure (higher = more similar)
                similarity = 1.0 / (1.0 + np.linalg.norm(query_vec - vec))
                similarities.append((i, similarity))

            # Sort by similarity descending and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_indices = [idx for idx, _ in similarities[:top_k]]

        return similar_indices[:top_k]


def cluster_users(
    user_series: List[pd.Series],
    n_clusters: int = 5
) -> UserClustering:
    """One-shot convenience function: create a UserClustering, fit it, and return it.

    Args:
        user_series: List of pandas Series, one per user.
        n_clusters: Number of clusters for KMeans.

    Returns:
        A fitted UserClustering object ready for predict() calls.
    """
    clusterer = UserClustering(n_clusters=n_clusters)  # Instantiate with defaults
    clusterer.fit(user_series)                          # Fit on all user data
    return clusterer

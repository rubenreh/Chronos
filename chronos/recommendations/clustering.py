"""User clustering for matching similar productivity profiles."""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from chronos.features.extractor import extract_behavioral_features


class UserClustering:
    """Cluster users by productivity patterns."""
    
    def __init__(
        self,
        n_clusters: int = 5,
        method: str = 'kmeans',
        normalize: bool = True
    ):
        """Initialize user clustering.
        
        Args:
            n_clusters: Number of clusters (for KMeans)
            method: Clustering method ('kmeans' or 'dbscan')
            normalize: Whether to normalize features
        """
        self.n_clusters = n_clusters
        self.method = method
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.clusterer = None
        self.feature_names_ = None
    
    def fit(self, user_series: List[pd.Series]):
        """Fit clustering model on user series.
        
        Args:
            user_series: List of time series, one per user
        """
        # Extract features for each user
        features_list = []
        for series in user_series:
            features = extract_behavioral_features(series)
            features_list.append(list(features.values()))
        
        if not features_list:
            raise ValueError("No features extracted from user series")
        
        self.feature_names_ = list(extract_behavioral_features(user_series[0]).keys())
        X = np.array(features_list)
        
        # Normalize if requested
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        # Fit clusterer
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        self.clusterer.fit(X)
    
    def predict(self, series: pd.Series) -> int:
        """Predict cluster for a single user series.
        
        Args:
            series: User's time series
        
        Returns:
            Cluster label
        """
        if self.clusterer is None:
            raise ValueError("Clustering model not fitted. Call fit() first.")
        
        features = extract_behavioral_features(series)
        X = np.array([list(features.values())])
        
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        
        return int(self.clusterer.predict(X)[0])
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers (for KMeans only)."""
        if self.method != 'kmeans' or self.clusterer is None:
            raise ValueError("Cluster centers only available for fitted KMeans")
        return self.clusterer.cluster_centers_
    
    def get_silhouette_score(self, user_series: List[pd.Series]) -> float:
        """Calculate silhouette score for clustering."""
        if self.clusterer is None:
            raise ValueError("Clustering model not fitted")
        
        features_list = []
        for series in user_series:
            features = extract_behavioral_features(series)
            features_list.append(list(features.values()))
        
        X = np.array(features_list)
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        
        labels = self.clusterer.labels_
        if len(set(labels)) < 2:
            return -1.0  # Not enough clusters for silhouette score
        
        return float(silhouette_score(X, labels))
    
    def find_similar_users(
        self,
        query_series: pd.Series,
        all_user_series: List[pd.Series],
        top_k: int = 5
    ) -> List[int]:
        """Find users with similar productivity profiles.
        
        Args:
            query_series: Query user's time series
            all_user_series: List of all user time series
            top_k: Number of similar users to return
        
        Returns:
            List of indices of similar users
        """
        query_cluster = self.predict(query_series)
        
        # Find all users in the same cluster
        similar_indices = []
        for i, series in enumerate(all_user_series):
            cluster = self.predict(series)
            if cluster == query_cluster:
                similar_indices.append(i)
        
        # If not enough in same cluster, find by feature similarity
        if len(similar_indices) < top_k:
            query_features = extract_behavioral_features(query_series)
            query_vec = np.array(list(query_features.values()))
            
            similarities = []
            for i, series in enumerate(all_user_series):
                features = extract_behavioral_features(series)
                vec = np.array(list(features.values()))
                similarity = 1.0 / (1.0 + np.linalg.norm(query_vec - vec))
                similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_indices = [idx for idx, _ in similarities[:top_k]]
        
        return similar_indices[:top_k]


def cluster_users(
    user_series: List[pd.Series],
    n_clusters: int = 5
) -> UserClustering:
    """Convenience function to cluster users."""
    clusterer = UserClustering(n_clusters=n_clusters)
    clusterer.fit(user_series)
    return clusterer


import numpy as np
import pandas as pd
import os

MODEL_DIR = "models"

class ART1:
    """
    ART1 Network for binary pattern clustering.
    """

    def __init__(self, num_features, max_categories=50, alpha=0.01, rho=0.8):
        self.num_features = num_features
        self.max_categories = max_categories
        self.alpha = alpha            # choice parameter
        self.rho = rho                # vigilance parameter

        # initialize category weight matrix (start fully active)
        self.W = np.ones((max_categories, num_features))  
        self.num_categories_used = 0

    @staticmethod
    def _and_operation(a, b):
        """Logical AND for binary vectors."""
        return np.minimum(a, b)

    def _choice_function(self, x):
        """
        Compute choice function T_j for all categories.
        """
        T = []
        for j in range(self.num_categories_used):
            numerator = np.sum(self._and_operation(x, self.W[j]))
            denominator = self.alpha + np.sum(self.W[j])
            T.append(numerator / denominator)

        return np.array(T)

    def _vigilance_test(self, x, j):
        """
        ART vigilance constraint test.
        """
        numerator = np.sum(self._and_operation(x, self.W[j]))
        denominator = np.sum(x)
        return (numerator / (denominator + 1e-9)) >= self.rho

    def _learn(self, x, j):
        """
        Update category weights using ART1 learning rule.
        """
        self.W[j] = self._and_operation(x, self.W[j])

    def fit(self, X):
        """
        Cluster all binary patterns X.
        X shape: (num_samples, num_features)
        Returns cluster assignments.
        """
        num_samples = X.shape[0]
        clusters = np.full(num_samples, -1)

        for i, x in enumerate(X):
            print(f"Processing sample {i+1}/{num_samples}...")

            # ---------- CATEGORY SELECTION ----------
            if self.num_categories_used == 0:
                # first category
                self._learn(x, 0)
                clusters[i] = 0
                self.num_categories_used = 1
                continue

            T = self._choice_function(x)
            indices = np.argsort(-T)  # descending order

            assigned = False
            for j in indices:
                if self._vigilance_test(x, j):
                    self._learn(x, j)
                    clusters[i] = j
                    assigned = True
                    break

            # ---------- NEW CATEGORY CREATION ----------
            if not assigned:
                if self.num_categories_used < self.max_categories:
                    j_new = self.num_categories_used
                    self.W[j_new] = x.copy()
                    clusters[i] = j_new
                    self.num_categories_used += 1
                else:
                    print("⚠ MAX CATEGORIES REACHED. Cannot add new cluster.")
                    clusters[i] = -1

        return clusters

    def save_results(self):
        """
        Save clusters + category weights.
        """
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        np.save(os.path.join(MODEL_DIR, "art1_weights.npy"), self.W)
        # Save metadata including number of categories
        metadata = {'num_categories_used': self.num_categories_used}
        np.save(os.path.join(MODEL_DIR, "art1_metadata.npy"), metadata)
        print("ART1 weights saved.")

    def load_results(self):
        self.W = np.load(os.path.join(MODEL_DIR, "art1_weights.npy"))
        # Load number of categories used
        metadata_path = os.path.join(MODEL_DIR, "art1_metadata.npy")
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
            self.num_categories_used = metadata['num_categories_used']
        print("ART1 weights loaded.")

    def predict(self, X):
        """
        Predict cluster assignments for input data without learning.
        X shape: (num_samples, num_features)
        Returns cluster assignments.
        """
        num_samples = X.shape[0]
        clusters = np.full(num_samples, -1)

        for i, x in enumerate(X):
            if self.num_categories_used == 0:
                clusters[i] = -1
                continue

            T = self._choice_function(x)
            indices = np.argsort(-T)  # descending order

            assigned = False
            for j in indices:
                if self._vigilance_test(x, j):
                    clusters[i] = j
                    assigned = True
                    break

            if not assigned:
                clusters[i] = -1

        return clusters

    def transform(self, X):
        """
        Transform input data to cluster-based one-hot encoded features.
        X shape: (num_samples, num_features)
        Returns: (num_samples, num_categories_used)
        """
        if self.num_categories_used == 0:
            raise ValueError("ART1 model not trained. Call fit() first.")
        
        clusters = self.predict(X)
        
        # Create one-hot encoded features
        features = np.zeros((len(X), self.num_categories_used))
        for i, cluster_id in enumerate(clusters):
            if cluster_id >= 0:
                features[i, cluster_id] = 1
        
        return features


def run_art1_clustering(input_file="student_binary.csv", alpha=0.01, rho=0.8, max_categories=500):
    """
    Load binary dataset, perform ART1 clustering, save model output.
    """
    path = os.path.join("data", "processed", input_file)
    df = pd.read_csv(path)
    X = df.values

    print("Running ART1 on dataset:", X.shape)

    art1 = ART1(num_features=X.shape[1], alpha=alpha, rho=rho, max_categories=max_categories)
    clusters = art1.fit(X)
    art1.save_results()

    np.save(os.path.join(MODEL_DIR, "art1_clusters.npy"), clusters)

    print("ART1 clustering complete.")
    print("Clusters used:", art1.num_categories_used)

    return clusters

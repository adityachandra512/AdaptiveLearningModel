import numpy as np
import pandas as pd
import os

MODEL_DIR = "models"   # folder to store Hebbian weights

class HebbianLearner:
    """
    Hebbian Learning for binary/bipolar feature correlation.
    Computes a correlation matrix W of size (features × features).
    """

    def __init__(self, bipolar=False):
        # if True: convert 0→-1 for bipolar representation
        self.bipolar = bipolar
        self.W = None

    def _convert_to_bipolar(self, X):
        """Convert 0/1 to -1/+1 if bipolar mode enabled."""
        return np.where(X == 0, -1, 1)

    def fit(self, X, track_evolution=False, evolution_interval=10):
        """
        Apply Hebbian learning on dataset X.
        X: numpy array (num_samples × num_features)
        """
        if self.bipolar:
            X = self._convert_to_bipolar(X)

        num_features = X.shape[1]

        # initialize weight matrix
        W = np.zeros((num_features, num_features))
        
        weight_history = []

        # Hebbian rule: W += x^T * x for each sample x
        for i, x in enumerate(X):
            x = x.reshape(-1, 1)     # column vector
            W += x @ x.T             # outer product
            
            if track_evolution and (i + 1) % evolution_interval == 0:
                weight_history.append(W.copy())

        # remove diagonal terms (no self-connections)
        np.fill_diagonal(W, 0)

        self.W = W
        
        if track_evolution:
            return W, weight_history
        return W

    def save_weights(self, filename="hebbian_weights.npy"):
        """Save the weight matrix to models/ folder."""
        if self.W is None:
            raise ValueError("Weights not computed. Call fit() first.")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        path = os.path.join(MODEL_DIR, filename)
        np.save(path, self.W)

        print(f"Hebbian weight matrix saved to: {path}")

    def load_weights(self, filename="hebbian_weights.npy"):
        """Load weight matrix from file."""
        path = os.path.join(MODEL_DIR, filename)
        self.W = np.load(path)
        print(f"Hebbian weight matrix loaded from: {path}")
        return self.W

    def transform(self, X):
        """
        Transform input data using learned Hebbian weights.
        
        Args:
            X: numpy array (num_samples × num_features)
            
        Returns:
            Transformed features (num_samples × num_features)
        """
        if self.W is None:
            raise ValueError("Weights not computed. Call fit() or load_weights() first.")
        
        X_input = X.copy()
        if self.bipolar:
            X_input = self._convert_to_bipolar(X_input)
        
        # Transform: X_transformed = X @ W
        return X_input @ self.W


def run_hebbian_learning(data=None, input_file="student_binary.csv", bipolar=False, save=True, track_evolution=False):
    """
    Load binary dataset, compute correlation matrix, save model.
    
    Args:
        data: Optional numpy array of binary data. If None, loads from input_file.
        input_file: CSV file to load if data is None.
        bipolar: Whether to use bipolar representation (-1/+1 instead of 0/1).
        save: Whether to save the weights to file.
        track_evolution: Whether to return the history of weight updates.
    
    Returns:
        Weight matrix as numpy array.
    """
    if data is None:
        path = os.path.join("data", "processed", input_file)
        df = pd.read_csv(path)
        X = df.values.astype(float)
    elif isinstance(data, pd.DataFrame):
        X = data.values.astype(float)
    else:
        X = data.astype(float)

    print("Running Hebbian Learning on dataset of shape:", X.shape)

    hebb = HebbianLearner(bipolar=bipolar)
    
    if track_evolution:
        W, history = hebb.fit(X, track_evolution=True)
    else:
        W = hebb.fit(X)
    
    if save:
        hebb.save_weights()

    print("Hebbian learning completed.")
    print("Weight matrix shape:", W.shape)

    if track_evolution:
        return W, history
    return W

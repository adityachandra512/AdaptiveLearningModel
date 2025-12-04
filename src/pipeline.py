import numpy as np
import pandas as pd
import os
import torch
from src.hebbian.hebbian_learning import HebbianLearner
from src.art1.art1_network import ART1
from src.mlp.model import MLP
from src.mlp.trainer import MLPTrainer

MODEL_DIR = "models"


class AdaptiveLearningPipeline:
    """
    Sequential pipeline: Hebbian → ART1 → MLP
    
    Flow:
    1. Hebbian Learning: Learns correlation patterns from binary data
    2. ART1 Clustering: Clusters Hebbian-transformed features
    3. MLP: Final prediction using ART1 cluster features
    """
    
    def __init__(self, 
                 hebbian_params=None,
                 art1_params=None,
                 mlp_params=None):
        """
        Initialize pipeline with model parameters.
        
        Args:
            hebbian_params: dict with keys: bipolar (default: False)
            art1_params: dict with keys: alpha, rho, max_categories
            mlp_params: dict with keys: hidden_size, learning_rate, epochs
        """
        self.hebbian_params = hebbian_params or {'bipolar': False}
        self.art1_params = art1_params or {'alpha': 0.01, 'rho': 0.8, 'max_categories': 500}
        self.mlp_params = mlp_params or {'hidden_size': 64, 'learning_rate': 0.001, 'epochs': 50}
        
        self.hebbian = None
        self.art1 = None
        self.mlp_trainer = None
        
        # Store intermediate data for inspection
        self.hebbian_features = None
        self.art1_features = None
        
    def fit(self, X_binary, y, verbose=True):
        """
        Train the complete pipeline.
        
        Args:
            X_binary: Binary input data (num_samples × num_features)
            y: Target labels (num_samples,)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history and intermediate outputs
        """
        if verbose:
            print("="*60)
            print("ADAPTIVE LEARNING PIPELINE - TRAINING")
            print("="*60)
        
        # ========== STAGE 1: HEBBIAN LEARNING ==========
        if verbose:
            print("\n[STAGE 1/3] Hebbian Learning")
            print("-" * 40)
        
        self.hebbian = HebbianLearner(bipolar=self.hebbian_params['bipolar'])
        W = self.hebbian.fit(X_binary)
        
        if verbose:
            print(f"✓ Hebbian weight matrix computed: {W.shape}")
        
        # Transform data using Hebbian weights
        self.hebbian_features = self.hebbian.transform(X_binary)
        
        if verbose:
            print(f"✓ Data transformed: {self.hebbian_features.shape}")
        
        # ========== STAGE 2: ART1 CLUSTERING ==========
        if verbose:
            print("\n[STAGE 2/3] ART1 Clustering")
            print("-" * 40)
        
        # Normalize Hebbian features to [0, 1] for ART1
        hebbian_normalized = self._normalize_for_art1(self.hebbian_features)
        
        self.art1 = ART1(
            num_features=hebbian_normalized.shape[1],
            alpha=self.art1_params['alpha'],
            rho=self.art1_params['rho'],
            max_categories=self.art1_params['max_categories']
        )
        
        clusters = self.art1.fit(hebbian_normalized)
        
        if verbose:
            print(f"✓ ART1 clustering complete")
            print(f"✓ Number of clusters formed: {self.art1.num_categories_used}")
        
        # Transform to cluster-based features
        self.art1_features = self.art1.transform(hebbian_normalized)
        
        if verbose:
            print(f"✓ Cluster features created: {self.art1_features.shape}")
        
        # ========== STAGE 3: MLP TRAINING ==========
        if verbose:
            print("\n[STAGE 3/3] MLP Training")
            print("-" * 40)
        
        # Train MLP on ART1 features
        input_size = self.art1_features.shape[1]
        
        # Simple train/val split
        split_idx = int(0.8 * len(self.art1_features))
        X_train = self.art1_features[:split_idx]
        X_val = self.art1_features[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        
        self.mlp_trainer = MLPTrainer(
            input_size=input_size,
            hidden_size=self.mlp_params['hidden_size'],
            output_size=1,
            learning_rate=self.mlp_params['learning_rate'],
            epochs=self.mlp_params['epochs']
        )
        
        history = self.mlp_trainer.train(X_train, y_train, X_val, y_val)
        
        if verbose:
            print(f"✓ MLP training complete")
            print("="*60)
        
        return {
            'hebbian_weights': W,
            'art1_clusters': clusters,
            'mlp_history': history,
            'hebbian_features_shape': self.hebbian_features.shape,
            'art1_features_shape': self.art1_features.shape
        }
    
    def predict(self, X_binary):
        """
        End-to-end prediction on new binary data.
        
        Args:
            X_binary: Binary input data (num_samples × num_features)
            
        Returns:
            Predictions from MLP
        """
        if self.hebbian is None or self.art1 is None or self.mlp_trainer is None:
            raise ValueError("Pipeline not trained. Call fit() first.")
        
        # Stage 1: Hebbian transform
        hebbian_features = self.hebbian.transform(X_binary)
        
        # Stage 2: ART1 transform
        hebbian_normalized = self._normalize_for_art1(hebbian_features)
        art1_features = self.art1.transform(hebbian_normalized)
        
        # Stage 3: MLP predict
        self.mlp_trainer.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(art1_features).to(self.mlp_trainer.device)
            predictions = self.mlp_trainer.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def _normalize_for_art1(self, X):
        """
        Normalize features to [0, 1] range for ART1.
        Uses min-max normalization.
        """
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        
        # Avoid division by zero
        denominator = X_max - X_min
        denominator[denominator == 0] = 1
        
        X_normalized = (X - X_min) / denominator
        
        # Binarize for ART1 (threshold at 0.5)
        return (X_normalized > 0.5).astype(float)
    
    def save_pipeline(self, prefix="pipeline"):
        """
        Save all models in the pipeline.
        
        Args:
            prefix: Prefix for saved model files
        """
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        # Save Hebbian
        if self.hebbian is not None:
            self.hebbian.save_weights(f"{prefix}_hebbian_weights.npy")
        
        # Save ART1
        if self.art1 is not None:
            np.save(os.path.join(MODEL_DIR, f"{prefix}_art1_weights.npy"), self.art1.W)
            metadata = {'num_categories_used': self.art1.num_categories_used}
            np.save(os.path.join(MODEL_DIR, f"{prefix}_art1_metadata.npy"), metadata)
        
        # Save MLP
        if self.mlp_trainer is not None:
            self.mlp_trainer.save_model(f"{prefix}_mlp_model.pt")
        
        # Save pipeline configuration
        config = {
            'hebbian_params': self.hebbian_params,
            'art1_params': self.art1_params,
            'mlp_params': self.mlp_params
        }
        np.save(os.path.join(MODEL_DIR, f"{prefix}_config.npy"), config)
        
        print(f"\n✓ Pipeline saved with prefix '{prefix}'")
    
    def load_pipeline(self, prefix="pipeline"):
        """
        Load all models in the pipeline.
        
        Args:
            prefix: Prefix of saved model files
        """
        # Load configuration
        config_path = os.path.join(MODEL_DIR, f"{prefix}_config.npy")
        config = np.load(config_path, allow_pickle=True).item()
        
        self.hebbian_params = config['hebbian_params']
        self.art1_params = config['art1_params']
        self.mlp_params = config['mlp_params']
        
        # Load Hebbian
        self.hebbian = HebbianLearner(bipolar=self.hebbian_params['bipolar'])
        self.hebbian.load_weights(f"{prefix}_hebbian_weights.npy")
        
        # Load ART1
        art1_weights = np.load(os.path.join(MODEL_DIR, f"{prefix}_art1_weights.npy"))
        art1_metadata = np.load(os.path.join(MODEL_DIR, f"{prefix}_art1_metadata.npy"), allow_pickle=True).item()
        
        self.art1 = ART1(
            num_features=art1_weights.shape[1],
            alpha=self.art1_params['alpha'],
            rho=self.art1_params['rho'],
            max_categories=self.art1_params['max_categories']
        )
        self.art1.W = art1_weights
        self.art1.num_categories_used = art1_metadata['num_categories_used']
        
        # Load MLP
        mlp_path = os.path.join(MODEL_DIR, f"{prefix}_mlp_model.pt")
        checkpoint = torch.load(mlp_path)
        
        self.mlp_trainer = MLPTrainer(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size'],
            learning_rate=self.mlp_params['learning_rate'],
            epochs=self.mlp_params['epochs']
        )
        self.mlp_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\n✓ Pipeline loaded with prefix '{prefix}'")


def run_pipeline(binary_file="student_binary.csv", 
                 normalized_file="student_normalized.csv",
                 target_col="G3",
                 save=True):
    """
    Run the complete Hebbian → ART1 → MLP pipeline.
    
    Args:
        binary_file: Binary data CSV file
        normalized_file: Normalized data CSV file (for target labels)
        target_col: Target column name
        save: Whether to save the trained pipeline
        
    Returns:
        Trained pipeline object
    """
    # Load binary data
    binary_path = os.path.join("data", "processed", binary_file)
    X_binary = pd.read_csv(binary_path).values
    
    # Load target labels from normalized data
    normalized_path = os.path.join("data", "processed", normalized_file)
    df_normalized = pd.read_csv(normalized_path)
    
    if target_col not in df_normalized.columns:
        target_col = df_normalized.columns[-1]
    
    y = df_normalized[target_col].values
    
    print(f"Loaded binary data: {X_binary.shape}")
    print(f"Loaded target labels: {y.shape}")
    
    # Create and train pipeline
    pipeline = AdaptiveLearningPipeline()
    results = pipeline.fit(X_binary, y, verbose=True)
    
    if save:
        pipeline.save_pipeline()
    
    return pipeline, results

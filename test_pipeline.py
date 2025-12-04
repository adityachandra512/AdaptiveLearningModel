"""
Test script to demonstrate the Hebbian → ART1 → MLP pipeline.
This script shows how to:
1. Load a trained pipeline
2. Make predictions on new data
3. Inspect intermediate outputs
"""

import numpy as np
import pandas as pd
import os
from src.pipeline import AdaptiveLearningPipeline

def test_pipeline_prediction():
    """Test the trained pipeline on sample data."""
    
    print("="*60)
    print("PIPELINE PREDICTION TEST")
    print("="*60)
    
    # Load binary test data
    binary_path = os.path.join("data", "processed", "student_binary.csv")
    X_binary = pd.read_csv(binary_path).values
    
    print(f"\nLoaded test data: {X_binary.shape}")
    
    # Load trained pipeline
    print("\nLoading trained pipeline...")
    pipeline = AdaptiveLearningPipeline()
    pipeline.load_pipeline(prefix="pipeline")
    
    # Make predictions on first 10 samples
    print("\nMaking predictions on first 10 samples...")
    X_test = X_binary[:10]
    predictions = pipeline.predict(X_test)
    
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred[0]:.4f}")
    
    # Inspect intermediate transformations
    print("\n" + "="*60)
    print("INTERMEDIATE TRANSFORMATIONS")
    print("="*60)
    
    # Hebbian transform
    hebbian_features = pipeline.hebbian.transform(X_test)
    print(f"\n1. Hebbian Transform:")
    print(f"   Input shape:  {X_test.shape}")
    print(f"   Output shape: {hebbian_features.shape}")
    print(f"   Sample output (first 5 features): {hebbian_features[0, :5]}")
    
    # ART1 transform
    hebbian_normalized = pipeline._normalize_for_art1(hebbian_features)
    art1_features = pipeline.art1.transform(hebbian_normalized)
    print(f"\n2. ART1 Transform:")
    print(f"   Input shape:  {hebbian_normalized.shape}")
    print(f"   Output shape: {art1_features.shape}")
    print(f"   Number of active clusters: {int(art1_features[0].sum())}")
    print(f"   Cluster assignment: {np.where(art1_features[0] == 1)[0]}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


def inspect_pipeline_models():
    """Inspect the saved pipeline models."""
    
    print("\n" + "="*60)
    print("PIPELINE MODEL INSPECTION")
    print("="*60)
    
    pipeline = AdaptiveLearningPipeline()
    pipeline.load_pipeline(prefix="pipeline")
    
    print(f"\n1. Hebbian Model:")
    print(f"   Weight matrix shape: {pipeline.hebbian.W.shape}")
    print(f"   Bipolar mode: {pipeline.hebbian.bipolar}")
    print(f"   Weight range: [{pipeline.hebbian.W.min():.2f}, {pipeline.hebbian.W.max():.2f}]")
    
    print(f"\n2. ART1 Model:")
    print(f"   Number of features: {pipeline.art1.num_features}")
    print(f"   Number of clusters used: {pipeline.art1.num_categories_used}")
    print(f"   Max categories: {pipeline.art1.max_categories}")
    print(f"   Vigilance (rho): {pipeline.art1.rho}")
    print(f"   Alpha: {pipeline.art1.alpha}")
    
    print(f"\n3. MLP Model:")
    print(f"   Input size: {pipeline.mlp_trainer.model.fc1.in_features}")
    print(f"   Hidden layer 1: {pipeline.mlp_trainer.model.fc1.out_features}")
    print(f"   Hidden layer 2: {pipeline.mlp_trainer.model.fc2.out_features}")
    print(f"   Output size: {pipeline.mlp_trainer.model.fc3.out_features}")
    print(f"   Device: {pipeline.mlp_trainer.device}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run prediction test
    test_pipeline_prediction()
    
    # Inspect models
    inspect_pipeline_models()

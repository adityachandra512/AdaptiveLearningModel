import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.mlp.trainer import run_mlp_training
from src.hebbian.hebbian_learning import run_hebbian_learning
from src.art1.art1_network import run_art1_clustering

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def plot_mlp_learning_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('MLP Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'mlp_learning_curve.png'))
    plt.close()
    print("Saved mlp_learning_curve.png")

def plot_hebbian_weights(W):
    plt.figure(figsize=(12, 10))
    # Plot only a subset if too large
    if W.shape[0] > 50:
        sns.heatmap(W[:50, :50], cmap='coolwarm', center=0)
        plt.title('Hebbian Weight Matrix (First 50 Features)')
    else:
        sns.heatmap(W, cmap='coolwarm', center=0)
        plt.title('Hebbian Weight Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'hebbian_weights.png'))
    plt.close()
    print("Saved hebbian_weights.png")

def plot_hebbian_evolution(history):
    # Plot snapshots
    snapshots = [0, len(history)//2, len(history)-1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, idx in enumerate(snapshots):
        if idx < len(history):
            w_snapshot = history[idx]
            if w_snapshot.shape[0] > 50:
                w_snapshot = w_snapshot[:50, :50]
                
            sns.heatmap(w_snapshot, ax=axes[i], cmap='coolwarm', center=0, cbar=False)
            axes[i].set_title(f'Step {idx*10}') # Assuming interval 10
            
    plt.suptitle('Hebbian Weight Evolution (First 50 Features)')
    plt.savefig(os.path.join(RESULTS_DIR, 'hebbian_evolution_snapshots.png'))
    plt.close()
    print("Saved hebbian_evolution_snapshots.png")

def plot_art1_clusters(clusters, data):
    # Bar chart
    unique, counts = np.unique(clusters, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts)
    plt.title('ART1 Cluster Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.savefig(os.path.join(RESULTS_DIR, 'art1_clusters.png'))
    plt.close()
    print("Saved art1_clusters.png")
    
    # PCA
    # Ensure data matches clusters length
    if len(data) != len(clusters):
        print(f"Warning: Data length {len(data)} != Clusters length {len(clusters)}")
        min_len = min(len(data), len(clusters))
        data = data[:min_len]
        clusters = clusters[:min_len]
        
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    # Filter out unassigned clusters (-1) if any
    mask = clusters != -1
    if np.sum(mask) > 0:
        scatter = plt.scatter(reduced[mask, 0], reduced[mask, 1], c=clusters[mask], cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
    
    # Plot unassigned as black x
    if np.sum(~mask) > 0:
        plt.scatter(reduced[~mask, 0], reduced[~mask, 1], c='black', marker='x', label='Unassigned')
        
    plt.title('ART1 Clusters (PCA Projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'art1_pca.png'))
    plt.close()
    print("Saved art1_pca.png")

def main():
    print("Starting Analysis...")
    
    # 1. MLP
    print("\n--- Analyzing MLP ---")
    try:
        _, mlp_history = run_mlp_training(return_history=True)
        plot_mlp_learning_curve(mlp_history)
    except Exception as e:
        print(f"Error analyzing MLP: {e}")
    
    # 2. Hebbian
    print("\n--- Analyzing Hebbian ---")
    try:
        W, hebb_history = run_hebbian_learning(track_evolution=True)
        plot_hebbian_weights(W)
        plot_hebbian_evolution(hebb_history)
    except Exception as e:
        print(f"Error analyzing Hebbian: {e}")
    
    # 3. ART1
    print("\n--- Analyzing ART1 ---")
    try:
        # Need to load data for PCA
        data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/student_binary.csv')
        df = pd.read_csv(data_path)
        X = df.values
        
        clusters = run_art1_clustering()
        plot_art1_clusters(clusters, X)
    except Exception as e:
        print(f"Error analyzing ART1: {e}")
    
    print("\nAnalysis Complete. Results saved to results/")

if __name__ == "__main__":
    main()

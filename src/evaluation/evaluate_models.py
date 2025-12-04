import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.mlp.trainer import MLPTrainer
from src.hebbian.hebbian_learning import HebbianLearner
from src.art1.art1_network import ART1

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def evaluate_mlp(X_train, X_test, y_train, y_test):
    print("Evaluating MLP...")
    trainer = MLPTrainer(input_size=X_train.shape[1])
    trainer.train(X_train, y_train)
    
    trainer.model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(trainer.device)
        outputs = trainer.model(X_tensor)
        predicted = (outputs.squeeze() > 0.5).float().cpu().numpy()
        
    return predicted

def evaluate_hebbian(X_train, X_test, y_train, y_test):
    print("Evaluating Hebbian...")
    # Augment training data with target
    y_train_reshaped = y_train.reshape(-1, 1)
    X_augmented = np.hstack([X_train, y_train_reshaped])
    
    hebb = HebbianLearner(bipolar=False) # Using binary 0/1
    W = hebb.fit(X_augmented)
    
    # The last column of W corresponds to the target's correlations
    # We want the weights from features to the target
    # W is (n_features+1) x (n_features+1)
    # Target index is -1
    w_target = W[:-1, -1] 
    
    # Predict: score = X @ w
    scores = X_test @ w_target
    
    # Thresholding: Since data is 0/1 and weights are correlations, 
    # positive score implies class 1, negative/zero implies class 0?
    # Or we can use the mean score as threshold.
    # Let's try 0 as threshold for now.
    predicted = (scores > 0).astype(float)
    
    return predicted

def evaluate_art1(X_train, X_test, y_train, y_test):
    print("Evaluating ART1...")
    # Train unsupervised
    art1 = ART1(num_features=X_train.shape[1], max_categories=100, rho=0.7)
    train_clusters = art1.fit(X_train)
    
    # Label clusters
    cluster_labels = {}
    unique_clusters = np.unique(train_clusters)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1: continue
        
        # Get indices of samples in this cluster
        indices = np.where(train_clusters == cluster_id)[0]
        # Get corresponding true labels
        labels = y_train[indices]
        # Majority vote
        counts = np.bincount(labels.astype(int))
        majority_label = np.argmax(counts)
        cluster_labels[cluster_id] = majority_label
        
    # Predict on test
    test_clusters = []
    # We need to expose a predict method or reuse internal logic
    # Since ART1 class doesn't have predict, we simulate it using the same logic
    # But wait, fit() modifies state. We need a predict_sample method.
    # We can manually do it here using the trained weights art1.W
    
    predictions = []
    majority_class = np.argmax(np.bincount(y_train.astype(int)))
    
    for x in X_test:
        # Find best matching category
        T = art1._choice_function(x)
        indices = np.argsort(-T)
        
        assigned_cluster = -1
        for j in indices:
            if art1._vigilance_test(x, j):
                assigned_cluster = j
                break
        
        if assigned_cluster != -1 and assigned_cluster in cluster_labels:
            predictions.append(cluster_labels[assigned_cluster])
        else:
            # Unassigned or new cluster (which we can't label) -> fallback
            predictions.append(majority_class)
            
    return np.array(predictions)

def calculate_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

def plot_confusion_matrices(cms, model_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (cm, name) in enumerate(zip(cms, model_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'))
    print("Saved confusion_matrices.png")

def main():
    # Load Data
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/student_binary.csv')
    df = pd.read_csv(data_path)
    
    # Target: Let's use 'G3_y' (final grade > 10 usually, but here it's binary)
    # If G3_y is not present, use last column
    target_col = 'G3_y' if 'G3_y' in df.columns else df.columns[-1]
    print(f"Target Column: {target_col}")
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    cms = []
    models = ['MLP', 'Hebbian', 'ART1']
    
    # 1. MLP
    y_pred_mlp = evaluate_mlp(X_train, X_test, y_train, y_test)
    results.append(calculate_metrics(y_test, y_pred_mlp, 'MLP'))
    cms.append(confusion_matrix(y_test, y_pred_mlp))
    
    # 2. Hebbian
    y_pred_hebb = evaluate_hebbian(X_train, X_test, y_train, y_test)
    results.append(calculate_metrics(y_test, y_pred_hebb, 'Hebbian'))
    cms.append(confusion_matrix(y_test, y_pred_hebb))
    
    # 3. ART1
    y_pred_art = evaluate_art1(X_train, X_test, y_train, y_test)
    results.append(calculate_metrics(y_test, y_pred_art, 'ART1'))
    cms.append(confusion_matrix(y_test, y_pred_art))
    
    # Save Metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'evaluation_metrics.csv'), index=False)
    print("\nEvaluation Metrics:")
    print(results_df)
    
    # Plot
    plot_confusion_matrices(cms, models)
    
    # Generate Report
    with open(os.path.join(RESULTS_DIR, 'evaluation_report.md'), 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Performance Metrics\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Analysis\n")
        f.write("- **MLP**: Supervised learning baseline.\n")
        f.write("- **Hebbian**: Simple correlation-based classifier.\n")
        f.write("- **ART1**: Unsupervised clustering adapted for classification.\n")
        f.write("\nCheck `confusion_matrices.png` for detailed error analysis.\n")
        
    print("Saved evaluation_report.md")

if __name__ == "__main__":
    main()

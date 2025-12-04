import os
# Set which CUDA devices to use (e.g., "0", "1,2,3", or "" for CPU only)
# Change this to select specific GPUs or use "" to force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use only GPU 0, change as needed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from src.mlp.model import MLP

MODEL_DIR = "/home/dgxuser16/NTL/norman/Aditya/AdaptiveLearningSystem/models"

class MLPTrainer:
    def __init__(self, input_size, hidden_size=64, output_size=1, learning_rate=0.001, epochs=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_size, hidden_size, output_size).to(self.device)
        self.criterion = nn.BCELoss() if output_size == 1 else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.train()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device) if self.model.fc3.out_features == 1 else torch.LongTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device) if self.model.fc3.out_features == 1 else torch.LongTensor(y_val).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"Training MLP on {self.device}...")
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.model.fc3.out_features == 1:
                    loss = self.criterion(outputs.squeeze(), batch_y)
                else:
                    loss = self.criterion(outputs, batch_y)
                    
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(loader)
            history['train_loss'].append(avg_train_loss)
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    if self.model.fc3.out_features == 1:
                        val_loss = self.criterion(val_outputs.squeeze(), y_val_tensor)
                    else:
                        val_loss = self.criterion(val_outputs, y_val_tensor)
                    history['val_loss'].append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                val_msg = f", Val Loss: {history['val_loss'][-1]:.4f}" if history['val_loss'] else ""
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_train_loss:.4f}{val_msg}")
                
        return history

    def save_model(self, filename="mlp_model.pt"):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        path = os.path.join(MODEL_DIR, filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.fc1.in_features,
            'hidden_size': self.model.fc1.out_features,
            'hidden_size2': self.model.fc2.out_features,
            'output_size': self.model.fc3.out_features
        }, path)

        print(f"MLP model saved to: {path}")


def run_mlp_training(input_file="student_normalized.csv", target_col="G3", return_history=False):
    """
    Train MLP on normalized dataset.
    """
    path = os.path.join("data", "processed", input_file)
    df = pd.read_csv(path)
    
    if target_col not in df.columns:
        # Fallback: use the last column
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # Simple split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train
    trainer = MLPTrainer(input_size=X.shape[1])
    history = trainer.train(X_train, y_train, X_val, y_val)
    trainer.save_model()
    
    if return_history:
        return trainer.model, history
    return trainer.model

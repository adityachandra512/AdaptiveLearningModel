import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for classification/regression.
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, task_type='classification'):
        super(MLP, self).__init__()
        self.task_type = task_type
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        if self.task_type == 'classification':
            if x.shape[1] == 1:
                return torch.sigmoid(x)
            else:
                return F.softmax(x, dim=1)
        return x

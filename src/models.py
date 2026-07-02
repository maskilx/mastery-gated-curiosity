import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class EnsembleMLP:
    def __init__(self, num_models=3, input_dim=6, hidden_dim=32, lr=0.01):
        self.num_models = num_models
        self.models = [SimpleMLP(input_dim, hidden_dim) for _ in range(num_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self.criterion = nn.MSELoss()
        
    def train_batch(self, X, Y):
        """
        X: tensor or array (batch_size, input_dim)
        Y: tensor or array (batch_size,)
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.float32).view(-1, 1)
        
        losses = []
        for i in range(self.num_models):
            self.optimizers[i].zero_grad()
            preds = self.models[i](X)
            loss = self.criterion(preds, Y)
            loss.backward()
            self.optimizers[i].step()
            losses.append(loss.item())
            
        return np.mean(losses)
        
    def predict(self, X):
        """
        Returns predictions for all models: (num_models, batch_size)
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = []
            for i in range(self.num_models):
                preds.append(self.models[i](X).view(-1).numpy())
        return np.array(preds)
        
    def predict_with_uncertainty(self, X):
        """
        Returns mean prediction and variance across ensemble.
        """
        preds = self.predict(X)
        mean_pred = np.mean(preds, axis=0)
        variance = np.var(preds, axis=0)
        return mean_pred, variance

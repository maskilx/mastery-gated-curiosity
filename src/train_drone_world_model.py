import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.drone_2d_env import Custom2DDroneEnv

class DroneWorldModel(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=64):
        super().__init__()
        # Input: state + action -> 8 dims
        # Output: delta state -> 6 dims
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Running normalization stats
        self.register_buffer('input_mean', torch.zeros(state_dim + action_dim))
        self.register_buffer('input_std', torch.ones(state_dim + action_dim))
        self.register_buffer('target_mean', torch.zeros(state_dim))
        self.register_buffer('target_std', torch.ones(state_dim))
        
    def update_stats(self, inputs, targets):
        self.input_mean = inputs.mean(dim=0)
        self.input_std = inputs.std(dim=0) + 1e-5
        self.target_mean = targets.mean(dim=0)
        self.target_std = targets.std(dim=0) + 1e-5
        
    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        delta_norm = self.net(x_norm)
        return delta_norm * self.target_std + self.target_mean


def collect_random_transitions(env, num_samples=10000):
    states = []
    actions = []
    next_states = []
    
    obs, _ = env.reset()
    collected = 0
    
    while collected < num_samples:
        # Action that tends to keep it somewhat in the air for longer episodes (0.3 to 0.7)
        # Random uniform [0, 1] crashes almost instantly, but we'll just collect whatever
        action = np.random.uniform(0.3, 0.7, size=(2,))
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Only collect safe transitions (not crashing)
        if not info.get('crashed', False):
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            collected += 1
            
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            
    return np.array(states), np.array(actions), np.array(next_states)


def smoke_test_world_model():
    print("Initializing environment...")
    env = Custom2DDroneEnv(mode='explore')
    
    print("Collecting 10000 random safe transitions...")
    states, actions, next_states = collect_random_transitions(env, 10000)
    deltas = next_states - states
    
    # Split into train/val
    indices = np.random.permutation(len(states))
    split = int(0.8 * len(states))
    
    train_idx, val_idx = indices[:split], indices[split:]
    
    inputs = np.concatenate([states, actions], axis=1)
    
    X_train = torch.FloatTensor(inputs[train_idx])
    Y_train = torch.FloatTensor(deltas[train_idx])
    
    X_val = torch.FloatTensor(inputs[val_idx])
    Y_val = torch.FloatTensor(deltas[val_idx])
    
    model = DroneWorldModel()
    model.update_stats(X_train, Y_train)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    print("Training world model...")
    epochs = 20
    batch_size = 64
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            batch_X = X_train[idx]
            batch_Y = Y_train[idx]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_X)
            
        train_mse = total_loss / len(X_train)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = loss_fn(val_pred, Y_val).item()
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Train MSE = {train_mse:.6f}, Val MSE = {val_mse:.6f}")
            
    # Compute per-dimension error
    model.eval()
    with torch.no_grad():
        final_val_pred = model(X_val)
        dim_mse = torch.mean((final_val_pred - Y_val)**2, dim=0).numpy()
        
    dims = ['x', 'z', 'vx', 'vz', 'theta', 'theta_dot']
    print("\nFinal Per-Dimension Validation MSE:")
    for d, mse in zip(dims, dim_mse):
        print(f"{d:10s}: {mse:.8f}")
        
    os.makedirs('results/drone', exist_ok=True)
    with open('results/drone/smoke_test_summary.md', 'w') as f:
        f.write("# Phase 5 World Model Smoke Test\n\n")
        f.write("Successfully collected 10000 safe random transitions and trained a deterministic MLP world model.\n\n")
        f.write(f"**Final Train MSE**: {train_mse:.6f}\n")
        f.write(f"**Final Val MSE**: {val_mse:.6f}\n\n")
        f.write("## Per-Dimension Error\n")
        for d, mse in zip(dims, dim_mse):
            f.write(f"- **{d}**: {mse:.8f}\n")
            
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_mean': model.input_mean,
        'input_std': model.input_std,
        'target_mean': model.target_mean,
        'target_std': model.target_std
    }, 'results/drone/world_model.pth')
    print("Saved world model to results/drone/world_model.pth")
            
    print("\nSmoke test successful! Saved summary to results/drone/smoke_test_summary.md")

if __name__ == '__main__':
    smoke_test_world_model()

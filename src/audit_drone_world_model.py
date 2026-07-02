import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.drone_2d_env import Custom2DDroneEnv
from src.train_drone_world_model import DroneWorldModel

def load_model(path='results/drone/world_model.pth'):
    checkpoint = torch.load(path, map_location='cpu')
    model = DroneWorldModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.input_mean = checkpoint['input_mean']
    model.input_std = checkpoint['input_std']
    model.target_mean = checkpoint['target_mean']
    model.target_std = checkpoint['target_std']
    model.eval()
    return model

def simulate_rollout(env, model, initial_state, action_fn, steps=50):
    """
    Compare true vs predicted rollout.
    action_fn(t) returns the action to take at step t.
    """
    true_states = [initial_state]
    pred_states = [initial_state]
    
    # Setup true env
    env.reset()
    env.state = np.copy(initial_state)
    
    # We will step both
    curr_pred_state = np.copy(initial_state)
    
    for t in range(steps):
        action = action_fn(t)
        
        # True step
        next_true_state, reward, term, trunc, info = env.step(action)
        true_states.append(next_true_state)
        
        # Pred step
        with torch.no_grad():
            inp = np.concatenate([curr_pred_state, action])
            inp_t = torch.FloatTensor(inp).unsqueeze(0)
            delta_pred = model(inp_t).squeeze(0).numpy()
            
        curr_pred_state = curr_pred_state + delta_pred
        # Normalize angle prediction to [-pi, pi]
        curr_pred_state[4] = (curr_pred_state[4] + np.pi) % (2 * np.pi) - np.pi
        pred_states.append(curr_pred_state)
        
        if term or trunc:
            break
            
    return np.array(true_states), np.array(pred_states)

def test_held_out_actions(model, env):
    test_cases = {
        'low_thrust': np.array([0.1, 0.1]),
        'near_hover': np.array([0.5, 0.5]),
        'high_thrust': np.array([0.9, 0.9]),
        'asymmetric_thrust': np.array([0.2, 0.8]),
        'recovery': np.array([0.9, 0.1])
    }
    
    results = {}
    for name, action in test_cases.items():
        # Start from typical initial state
        state, _ = env.reset()
        env.state = np.array([0.0, 1.0, 0.0, 0.0, 0.1, 0.0]) # Hover-ish with slight tilt
        
        next_true, _, _, _, _ = env.step(action)
        
        with torch.no_grad():
            inp = np.concatenate([np.array([0.0, 1.0, 0.0, 0.0, 0.1, 0.0]), action])
            inp_t = torch.FloatTensor(inp).unsqueeze(0)
            delta_pred = model(inp_t).squeeze(0).numpy()
            next_pred = np.array([0.0, 1.0, 0.0, 0.0, 0.1, 0.0]) + delta_pred
            
        error = np.mean((next_true - next_pred)**2)
        results[name] = error
        
    return results

def test_near_boundary_states(model, env):
    # State format: [x, z, vx, vz, theta, theta_dot]
    test_cases = {
        'low_altitude': np.array([0.0, 0.05, 0.0, -1.0, 0.0, 0.0]),
        'high_tilt': np.array([0.0, 1.0, 0.0, 0.0, 1.4, 0.5]),
        'high_velocity': np.array([0.0, 1.0, 10.0, -5.0, 0.0, 0.0]),
        'high_angular_vel': np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])
    }
    
    results = {}
    action = np.array([0.5, 0.5])
    
    for name, state in test_cases.items():
        env.reset()
        env.state = np.copy(state)
        
        next_true, _, _, _, _ = env.step(action)
        
        with torch.no_grad():
            inp = np.concatenate([state, action])
            inp_t = torch.FloatTensor(inp).unsqueeze(0)
            delta_pred = model(inp_t).squeeze(0).numpy()
            next_pred = state + delta_pred
            
        error = np.mean((next_true - next_pred)**2)
        results[name] = error
        
    return results

def test_safety_prediction(model, env):
    # Predict transitions that cross crash boundaries
    test_cases = {
        'ground_crash': (np.array([0.0, 0.1, 0.0, -10.0, 0.0, 0.0]), np.array([0.0, 0.0])), # falling fast
        'flip_crash': (np.array([0.0, 1.0, 0.0, 0.0, 1.5, 5.0]), np.array([0.1, 0.9])) # tilting fast
    }
    
    results = {}
    for name, (state, action) in test_cases.items():
        env.reset()
        env.state = np.copy(state)
        next_true, _, term, trunc, info = env.step(action)
        true_crashed = info.get('crashed', False)
        
        with torch.no_grad():
            inp = np.concatenate([state, action])
            inp_t = torch.FloatTensor(inp).unsqueeze(0)
            delta_pred = model(inp_t).squeeze(0).numpy()
            next_pred = state + delta_pred
            next_pred[4] = (next_pred[4] + np.pi) % (2 * np.pi) - np.pi
            
            pred_crashed = (next_pred[1] < 0.0) or (abs(next_pred[4]) > (np.pi / 2.0))
            
        results[name] = {
            'TrueCrash': true_crashed,
            'PredCrash': bool(pred_crashed),
            'Match': true_crashed == pred_crashed
        }
        
    return results

def run_audit():
    os.makedirs('results/drone', exist_ok=True)
    model = load_model()
    env = Custom2DDroneEnv(mode='explore')
    
    # 1. Multi-step rollout
    initial_state = np.array([0.0, 1.0, 0.0, 0.0, 0.1, 0.0])
    # Mild recovery action policy: try to right the drone
    def recovery_policy(t):
        if t < 10:
            return np.array([0.6, 0.4])
        return np.array([0.5, 0.5])
        
    true_traj, pred_traj = simulate_rollout(env, model, initial_state, recovery_policy, steps=50)
    
    # Plot trajectories
    dims = ['x', 'z', 'vx', 'vz', 'theta', 'theta_dot']
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    error_growth = {5: [], 10: [], 25: [], 50: []}
    
    for i, dim_name in enumerate(dims):
        axes[i].plot(true_traj[:, i], label='True', color='blue')
        axes[i].plot(pred_traj[:, i], label='Pred', color='red', linestyle='--')
        axes[i].set_title(dim_name)
        axes[i].legend()
        
        # Calculate error at specific horizons
        for h in [5, 10, 25, 50]:
            if h < len(true_traj):
                err = np.mean((true_traj[:h, i] - pred_traj[:h, i])**2)
                error_growth[h].append(err)
                
    plt.tight_layout()
    plt.savefig('results/drone/world_model_rollout.png')
    plt.close()
    
    # 2. Held-out action regimes
    held_out_errs = test_held_out_actions(model, env)
    
    # 3. Near-boundary states
    boundary_errs = test_near_boundary_states(model, env)
    
    # 4. Safety prediction
    safety_preds = test_safety_prediction(model, env)
    
    # Save Summary
    with open('results/drone/world_model_robustness_summary.md', 'w') as f:
        f.write("# Phase 5A: World Model Robustness Audit\n\n")
        
        f.write("## 1. Multi-Step Error Growth\n")
        f.write("Mean Squared Error over rollout horizons (for dimensions: x, z, vx, vz, theta, theta_dot):\n")
        f.write("| Horizon | x | z | vx | vz | theta | theta_dot |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for h in [5, 10, 25, 50]:
            if h in error_growth and len(error_growth[h]) == 6:
                errs = error_growth[h]
                f.write(f"| {h} steps | {errs[0]:.6f} | {errs[1]:.6f} | {errs[2]:.6f} | {errs[3]:.6f} | {errs[4]:.6f} | {errs[5]:.6f} |\n")
        
        f.write("\n## 2. Held-Out Action Regimes (1-Step MSE)\n")
        for k, v in held_out_errs.items():
            f.write(f"- **{k}**: {v:.8f}\n")
            
        f.write("\n## 3. Near-Boundary States (1-Step MSE)\n")
        for k, v in boundary_errs.items():
            f.write(f"- **{k}**: {v:.8f}\n")
            
        f.write("\n## 4. Safety Prediction\n")
        for k, v in safety_preds.items():
            f.write(f"- **{k}**: True Crash = {v['TrueCrash']}, Pred Crash = {v['PredCrash']} (Match: {v['Match']})\n")
            
        f.write("\n![Rollout Plot](/Users/adimaskil/.gemini/antigravity/brain/1ca23ce5-45a9-4c1c-addf-8255e229008d/world_model_rollout.png)\n")
        
    print("Audit complete! Saved to results/drone/world_model_robustness_summary.md")

if __name__ == '__main__':
    run_audit()

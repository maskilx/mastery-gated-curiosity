import os
import numpy as np
from stable_baselines3 import PPO
from src.drone_2d_env import Custom2DDroneEnv
from src.drone_env_wrapper import SafetyGatedMasteryCuriosityWrapper

def run_smoke_test():
    print("Running Phase 5C Smoke Test...")
    
    # Initialize environment
    base_env = Custom2DDroneEnv(mode='explore')
    env = SafetyGatedMasteryCuriosityWrapper(base_env)
    
    # Train PPO for a short time
    model = PPO('MlpPolicy', env, verbose=1, n_steps=1024, batch_size=256)
    
    print("Starting PPO training (5000 steps)...")
    model.learn(total_timesteps=5000)
    print("Training finished without crashing out of loop!")
    
    # Access the tracker
    tracker = env.tracker
    num_visited = len(tracker.regimes)
    
    counts = {
        'unexplored': 0,
        'learning': 0,
        'mastered': 0,
        'blocked_unsafe': 0,
        'blocked_no_progress': 0
    }
    
    for k, regime in tracker.regimes.items():
        counts[regime['status']] += 1
        
    print(f"\nTotal visited regimes: {num_visited}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
        
    print("\nSample Regimes:")
    i = 0
    for k, v in tracker.regimes.items():
        print(f"  {k} -> visits: {v['visit_count']}, error: {v['error_ema']:.4f}, lp: {v['lp_ema']:.4f}, status: {v['status']}")
        i += 1
        if i >= 5:
            break
            
if __name__ == '__main__':
    run_smoke_test()

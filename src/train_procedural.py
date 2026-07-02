import os
import yaml
import copy
from stable_baselines3 import PPO
from .rl_env import CuriosityEnv
from .train_rl import RewardLoggerCallback

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_procedural(total_timesteps=200000, smoke_test=False):
    base_config = load_config()
    os.makedirs("results/procedural", exist_ok=True)
    
    config = copy.deepcopy(base_config)
    
    # Enable procedural generation for training
    config['environment']['procedural'] = True
    config['environment']['difficulty'] = 'medium'
    config['environment']['max_regions'] = 10
    
    # Use Variant B reward (Best from Phase 3B)
    config['rl'] = {
        'step_cost': 0.0002,
        'newly_mastered_bonus': 0.1,
        'noisy_region_penalty': 0.02,
        'false_mastery_penalty': 0.2,
        'false_blocked_penalty': 0.2,
        'learnable_difficulty_bonus': 0.0,
        'inactive_region_penalty': -0.05
    }
    
    env = CuriosityEnv(config, seed=42)
    model = PPO("MlpPolicy", env, verbose=1)
    callback = RewardLoggerCallback()
    
    if smoke_test:
        print("Running smoke test (1000 timesteps)...")
        model.learn(total_timesteps=1000, callback=callback)
        print("Smoke test passed.")
        return
        
    print("Running full procedural PPO training...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("results/procedural/ppo_procedural.zip")
    print("Training complete. Model saved to results/procedural/ppo_procedural.zip")

if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    train_procedural(total_timesteps=200000, smoke_test=smoke)

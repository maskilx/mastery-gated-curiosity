import os
import yaml
import copy
from stable_baselines3 import PPO
from .rl_env import CuriosityEnv
from .train_rl import RewardLoggerCallback

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_fixed_for_procedural():
    base_config = load_config()
    os.makedirs("results/rl", exist_ok=True)
    
    config = copy.deepcopy(base_config)
    config['environment']['procedural'] = False
    config['environment']['max_regions'] = 10
    
    config['rl'] = {
        'step_cost': 0.0002,
        'newly_mastered_bonus': 0.1,
        'noisy_region_penalty': 0.02,
        'false_mastery_penalty': 0.2,
        'false_blocked_penalty': 0.2,
        'learnable_difficulty_bonus': 0.0,
        'inactive_region_penalty': -0.05
    }
    
    print("Retraining PPO_Fixed (Variant B) with max_regions=10...")
    env = CuriosityEnv(config, seed=42)
    model = PPO("MlpPolicy", env, verbose=0)
    callback = RewardLoggerCallback()
    
    model.learn(total_timesteps=200000, callback=callback)
    model.save("results/rl/ppo_variant_b.zip")
    print("Saved to results/rl/ppo_variant_b.zip")

if __name__ == "__main__":
    train_fixed_for_procedural()

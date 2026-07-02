import os
import yaml
import copy
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from .rl_env import CuriosityEnv
from .train_rl import RewardLoggerCallback

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_phase3b():
    base_config = load_config()
    os.makedirs("results/rl", exist_ok=True)
    
    variants = {
        'ppo_variant_a': {
            'step_cost': 0.001,
            'newly_mastered_bonus': 0.05,
            'noisy_region_penalty': 0.02,
            'false_mastery_penalty': 0.2,
            'false_blocked_penalty': 0.2,
            'learnable_difficulty_bonus': 0.0
        },
        'ppo_variant_b': {
            'step_cost': 0.0002,
            'newly_mastered_bonus': 0.1,
            'noisy_region_penalty': 0.02,
            'false_mastery_penalty': 0.2,
            'false_blocked_penalty': 0.2,
            'learnable_difficulty_bonus': 0.0
        },
        'ppo_variant_c': {
            'step_cost': 0.0002,
            'newly_mastered_bonus': 0.1,
            'noisy_region_penalty': 0.02,
            'false_mastery_penalty': 0.2,
            'false_blocked_penalty': 0.2,
            'learnable_difficulty_bonus': 0.01
        }
    }
    
    for v_name, v_rl_conf in variants.items():
        print(f"\n--- Training {v_name} ---")
        config = copy.deepcopy(base_config)
        config['rl'] = v_rl_conf
        
        env = CuriosityEnv(config, seed=42)
        model = PPO("MlpPolicy", env, verbose=0)
        callback = RewardLoggerCallback()
        
        model.learn(total_timesteps=200000, callback=callback)
        model.save(f"results/rl/{v_name}.zip")
        
        plt.figure(figsize=(10, 5))
        plt.plot(callback.episode_rewards)
        plt.title(f"{v_name} Episode Reward over Training")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(f"results/rl/{v_name}_reward.png")
        plt.close()

if __name__ == "__main__":
    train_phase3b()

import os
import yaml
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .rl_env import CuriosityEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0
        
    def _on_step(self) -> bool:
        self.current_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_rl(smoke_test=False):
    config = load_config()
    os.makedirs("results/rl", exist_ok=True)
    
    env = CuriosityEnv(config, seed=42)
    
    model = PPO("MlpPolicy", env, verbose=1)
    callback = RewardLoggerCallback()
    
    timesteps = 1000 if smoke_test else 200000
    
    print(f"Starting PPO training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=callback)
    
    if not smoke_test:
        model.save("results/rl/ppo_curiosity_region.zip")
        print("Training complete. Model saved.")
        
        plt.figure(figsize=(10, 5))
        plt.plot(callback.episode_rewards)
        plt.title("PPO Episode Reward over Training")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig("results/rl/ppo_training_reward.png")
        plt.close()
    else:
        print("Smoke test complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke test")
    args = parser.parse_args()
    
    train_rl(smoke_test=args.smoke)

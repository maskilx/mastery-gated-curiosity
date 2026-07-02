import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from src.drone_2d_env import Custom2DDroneEnv

def train_downstream_hover(timesteps=500_000):
    os.makedirs('results/drone/models', exist_ok=True)
    os.makedirs('results/drone/logs', exist_ok=True)
    
    env_scratch = make_vec_env(lambda: Custom2DDroneEnv(mode='hover'), n_envs=1)
    env_eval = make_vec_env(lambda: Custom2DDroneEnv(mode='hover'), n_envs=1)
    
    # EvalCallback will save the best model and log the learning curve to a npz file
    def get_eval_callback(name):
        return EvalCallback(env_eval, best_model_save_path=f'results/drone/models/{name}_best',
                            log_path=f'results/drone/logs/{name}', eval_freq=10000,
                            deterministic=True, render=False)
                            
    # 1. PPO_Scratch
    print("Training PPO_Scratch...")
    model_scratch = PPO('MlpPolicy', env_scratch, verbose=1)
    model_scratch.learn(total_timesteps=timesteps, callback=get_eval_callback('scratch'))
    model_scratch.save("results/drone/models/ppo_scratch_hover")
    
    # 2. PPO_SurvivalPretrain
    print("Fine-tuning PPO_SurvivalPretrain...")
    model_surv = PPO.load("results/drone/models/ppo_survival_pretrained.zip", env=env_scratch)
    model_surv.learn(total_timesteps=timesteps, callback=get_eval_callback('survival'))
    model_surv.save("results/drone/models/ppo_survival_hover")
    
    # 3. PPO_CuriosityPretrain
    print("Fine-tuning PPO_CuriosityPretrain...")
    model_cur = PPO.load("results/drone/models/ppo_curiosity_pretrained.zip", env=env_scratch)
    model_cur.learn(total_timesteps=timesteps, callback=get_eval_callback('curiosity'))
    model_cur.save("results/drone/models/ppo_curiosity_hover")
    
    # 4. PPO_SurvivalCuriosityPretrain
    print("Fine-tuning PPO_SurvivalCuriosityPretrain...")
    model_surv_cur = PPO.load("results/drone/models/ppo_survival_curiosity_pretrained.zip", env=env_scratch)
    model_surv_cur.learn(total_timesteps=timesteps, callback=get_eval_callback('survival_curiosity'))
    model_surv_cur.save("results/drone/models/ppo_survival_curiosity_hover")
    
    # 5. PPO_SurvivalCuriosityMasteryPretrain
    print("Fine-tuning PPO_SurvivalCuriosityMasteryPretrain...")
    model_mastery = PPO.load("results/drone/models/ppo_safety_mastery_pretrained.zip", env=env_scratch)
    model_mastery.learn(total_timesteps=timesteps, callback=get_eval_callback('safety_mastery'))
    model_mastery.save("results/drone/models/ppo_safety_mastery_hover")
    
    print("Downstream hover training complete!")

if __name__ == '__main__':
    train_downstream_hover(timesteps=300_000)

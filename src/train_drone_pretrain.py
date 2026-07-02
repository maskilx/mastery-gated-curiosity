import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from src.drone_2d_env import Custom2DDroneEnv
from src.drone_env_wrapper import (
    SurvivalPretrainWrapper, 
    CuriosityPretrainWrapper,
    SurvivalCuriosityPretrainWrapper,
    SafetyGatedMasteryCuriosityWrapper
)

def make_survival_env():
    return SurvivalPretrainWrapper(Custom2DDroneEnv(mode='explore'))

def make_curiosity_env():
    return CuriosityPretrainWrapper(Custom2DDroneEnv(mode='explore'))

def make_survival_curiosity_env():
    return SurvivalCuriosityPretrainWrapper(Custom2DDroneEnv(mode='explore'))
    
def make_safety_mastery_env():
    return SafetyGatedMasteryCuriosityWrapper(Custom2DDroneEnv(mode='explore'))

def train_pretrain_models(timesteps=500_000):
    os.makedirs('results/drone/models', exist_ok=True)
    
    print("Training PPO_SurvivalPretrain...")
    # Use make_vec_env to get Monitor logging for free
    env_surv = make_vec_env(make_survival_env, n_envs=1)
    model_surv = PPO('MlpPolicy', env_surv, verbose=1)
    model_surv.learn(total_timesteps=timesteps)
    model_surv.save("results/drone/models/ppo_survival_pretrained")
    
    print("Training PPO_CuriosityPretrain...")
    env_cur = make_vec_env(make_curiosity_env, n_envs=1)
    model_cur = PPO('MlpPolicy', env_cur, verbose=1)
    model_cur.learn(total_timesteps=timesteps)
    model_cur.save("results/drone/models/ppo_curiosity_pretrained")
    
    print("Training PPO_SurvivalCuriosityPretrain...")
    env_surv_cur = make_vec_env(make_survival_curiosity_env, n_envs=1)
    model_surv_cur = PPO('MlpPolicy', env_surv_cur, verbose=1)
    model_surv_cur.learn(total_timesteps=timesteps)
    model_surv_cur.save("results/drone/models/ppo_survival_curiosity_pretrained")
    
    print("Training PPO_SurvivalCuriosityMasteryPretrain...")
    env_mastery = make_vec_env(make_safety_mastery_env, n_envs=1)
    model_mastery = PPO('MlpPolicy', env_mastery, verbose=1)
    model_mastery.learn(total_timesteps=timesteps)
    model_mastery.save("results/drone/models/ppo_safety_mastery_pretrained")
    
    print("Pretraining complete!")

if __name__ == '__main__':
    # 200k steps is usually enough for a 6D state space MLP policy to learn something reasonable
    train_pretrain_models(timesteps=200_000)

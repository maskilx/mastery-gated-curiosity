import os
import yaml
import copy
from sb3_contrib import MaskablePPO
from src.rl_env import CuriosityEnv, CuriosityEnvNoValObs, CuriosityEnvNoValObs_NoStatus
from src.train_rl import RewardLoggerCallback
from src.policy import RegionSetPolicy

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_curriculum(smoke_test=False, use_novalobs=False, use_nostatus=False):
    base_config = load_config()
    os.makedirs("results/procedural", exist_ok=True)
    
    config = copy.deepcopy(base_config)
    config['environment']['procedural'] = True
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
    
    stages = [
        {'name': 'Stage 1', 'difficulty': 'easy', 'fixed_num_active': 5, 'timesteps': 1000 if smoke_test else 100000},
        {'name': 'Stage 2', 'difficulty': 'medium', 'fixed_num_active': 5, 'timesteps': 1000 if smoke_test else 100000},
        {'name': 'Stage 3', 'difficulty': 'medium', 'fixed_num_active': None, 'timesteps': 1000 if smoke_test else 150000}
    ]
    
    print("--- Training PPO Set Policy ---")
    model_set = None
    for stage in stages:
        print(f"Starting {stage['name']}...")
        stage_config = copy.deepcopy(config)
        stage_config['environment']['difficulty'] = stage['difficulty']
        stage_config['environment']['fixed_num_active'] = stage['fixed_num_active']
        
        if use_nostatus:
            env = CuriosityEnvNoValObs_NoStatus(stage_config, seed=42)
        elif use_novalobs:
            env = CuriosityEnvNoValObs(stage_config, seed=42)
        else:
            env = CuriosityEnv(stage_config, seed=42)
        
        if model_set is None:
            model_set = MaskablePPO(RegionSetPolicy, env, verbose=0)
        else:
            model_set.set_env(env)
            
        callback = RewardLoggerCallback()
        model_set.learn(total_timesteps=stage['timesteps'], callback=callback)
        
    if not smoke_test:
        if use_nostatus:
            name = "ppo_set_policy_novalobs_nostatus.zip"
        elif use_novalobs:
            name = "ppo_set_policy_novalobs.zip"
        else:
            name = "ppo_set_policy.zip"
            
        model_set.save(f"results/procedural/{name}")
        print(f"Saved {name}")
        
    print("--- Training PPO Flat MLP ---")
    model_flat = None
    for stage in stages:
        print(f"Starting {stage['name']}...")
        stage_config = copy.deepcopy(config)
        stage_config['environment']['difficulty'] = stage['difficulty']
        stage_config['environment']['fixed_num_active'] = stage['fixed_num_active']
        
        if use_nostatus:
            env = CuriosityEnvNoValObs_NoStatus(stage_config, seed=42)
        elif use_novalobs:
            env = CuriosityEnvNoValObs(stage_config, seed=42)
        else:
            env = CuriosityEnv(stage_config, seed=42)
        
        if model_flat is None:
            model_flat = MaskablePPO("MlpPolicy", env, verbose=0)
        else:
            model_flat.set_env(env)
            
        callback = RewardLoggerCallback()
        model_flat.learn(total_timesteps=stage['timesteps'], callback=callback)
        
    if not smoke_test:
        if use_nostatus:
            name = "ppo_flat_policy_novalobs_nostatus.zip"
        elif use_novalobs:
            name = "ppo_flat_policy_novalobs.zip"
        else:
            name = "ppo_flat_policy.zip"
            
        model_flat.save(f"results/procedural/{name}")
        print(f"Saved {name}")

    print("Curriculum training complete.")

if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    novalobs = "--novalobs" in sys.argv
    nostatus = "--no-status" in sys.argv
    train_curriculum(smoke_test=smoke, use_novalobs=novalobs, use_nostatus=nostatus)

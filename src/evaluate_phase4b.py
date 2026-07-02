import os
import yaml
import copy
import pandas as pd
import numpy as np
from sb3_contrib import MaskablePPO

from .world_generator import ProceduralEnvironment
from .agent import MasteryGatedHypothesisAgent
from .baselines import RandomAgent, HighestErrorAgent, LearningProgressAgent
from .rl_env import CuriosityEnv

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class RLAgentWrapper:
    def __init__(self, model):
        self.model = model
        
    def select_action(self, active_mask):
        # We need to construct a CuriosityEnv-like observation, but here we just take
        # the action mask and pass it to predict
        action, _ = self.model.predict(self.obs, action_masks=active_mask.astype(bool), deterministic=True)
        return action
        
    def set_obs(self, obs):
        self.obs = obs

def evaluate_agent(agent_type, env_type, num_seeds=10, max_steps=1000, ppo_path=None, ppo_env=None):
    base_config = load_config()
    config = copy.deepcopy(base_config)
    config['environment']['procedural'] = True
    config['environment']['max_regions'] = 10
    
    if env_type == 'ID_Medium':
        config['environment']['difficulty'] = 'medium'
    elif env_type == 'OOD_Hard':
        config['environment']['difficulty'] = 'hard'
        
    results = []
    
    for seed in range(num_seeds):
        env = ProceduralEnvironment(difficulty=config['environment']['difficulty'], max_regions=10, seed=seed)
        
        if agent_type == 'PPO_FlatMLP' or agent_type == 'PPO_SetPolicy':
            model = MaskablePPO.load(ppo_path)
            agent = RLAgentWrapper(model)
            ppo_env.env = env # override inner env
            ppo_env.reset(seed=seed)
        elif agent_type == 'Random':
            agent = RandomAgent(10, config, seed=seed)
        elif agent_type == 'HighestError':
            agent = HighestErrorAgent(10, config, seed=seed)
        elif agent_type == 'LearningProgress':
            agent = LearningProgressAgent(10, config, seed=seed)
        elif agent_type == 'MasteryGatedHypothesis':
            agent = MasteryGatedHypothesisAgent(10, config, seed=seed)
            
        for step in range(max_steps):
            if agent_type.startswith('PPO'):
                obs = ppo_env._get_obs()
                agent.set_obs(obs)
                action = agent.select_action(env.active_mask)
                _, _, _, _, _ = ppo_env.step(action)
            else:
                action = agent.select_action(active_mask=env.active_mask)
                if agent_type == 'MasteryGatedHypothesis':
                    x = agent.select_x_for_region(action)
                else:
                    x = env.rng.uniform(-1, 1)
                
                y = env.measure(action, x)
                agent.record(action, x, y)
                agent.train(env)
                
        tracker = ppo_env.agent.tracker if agent_type.startswith('PPO') else agent.tracker
        
        # Calculate metrics
        active_indices = np.where(env.active_mask == 1.0)[0]
        learnable_indices = [r for r in active_indices if env.regions[r]['family'] != 'noisy_unlearnable']
        noisy_indices = [r for r in active_indices if env.regions[r]['family'] == 'noisy_unlearnable']
        inactive_indices = np.where(env.active_mask == 0.0)[0]
        
        if len(learnable_indices) > 0:
            val_error = np.mean([tracker.val_error[r] for r in learnable_indices])
            mastered_count = sum([1 for r in learnable_indices if tracker.status[r] == 'mastered'])
            false_blocked = sum([1 for r in learnable_indices if tracker.status[r] == 'blocked_noisy'])
        else:
            val_error = 0.0
            mastered_count = 0
            false_blocked = 0
            
        if len(noisy_indices) > 0:
            false_mastery = sum([1 for r in noisy_indices if tracker.status[r] == 'mastered'])
            noisy_samples = sum([tracker.num_samples[r] for r in noisy_indices])
        else:
            false_mastery = 0
            noisy_samples = 0
            
        inactive_samples = sum([tracker.num_samples[r] for r in inactive_indices])
        
        results.append({
            'Env': env_type,
            'Agent': agent_type,
            'Seed': seed,
            'ActiveRegions': env.num_active,
            'ValError': val_error,
            'SamplesWastedInactive': inactive_samples,
            'SamplesWastedNoisy': noisy_samples,
            'MasteredLearnable': mastered_count,
            'FalseMastery': false_mastery,
            'FalseBlocked': false_blocked
        })
        
    return results

def evaluate_phase4b():
    print("Starting Phase 4B Evaluation...")
    all_results = []
    
    agents = [
        'MasteryGatedHypothesis',
        'Random',
        'HighestError',
        'LearningProgress',
        'PPO_FlatMLP',
        'PPO_SetPolicy'
    ]
    envs = ['ID_Medium', 'OOD_Hard']
    
    config = load_config()
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
    ppo_env = CuriosityEnv(config, seed=42)
    
    for env_type in envs:
        print(f"--- Evaluating on {env_type} Worlds ---")
        for agent in agents:
            print(f"Evaluating {agent}...")
            ppo_path = None
            if agent == 'PPO_FlatMLP':
                ppo_path = "results/procedural/ppo_flat_policy.zip"
            elif agent == 'PPO_SetPolicy':
                ppo_path = "results/procedural/ppo_set_policy.zip"
                
            res = evaluate_agent(agent, env_type, ppo_path=ppo_path, ppo_env=ppo_env)
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    
    os.makedirs("results/procedural", exist_ok=True)
    df.to_csv("results/procedural/phase4b_results_raw.csv", index=False)
    
    summary = df.groupby(['Env', 'Agent']).agg({
        'ValError': ['mean', 'std'],
        'SamplesWastedInactive': 'mean',
        'SamplesWastedNoisy': 'mean',
        'MasteredLearnable': 'mean',
        'FalseMastery': 'mean',
        'FalseBlocked': 'mean'
    }).round(4)
    
    summary_by_region = df.groupby(['Env', 'ActiveRegions', 'Agent']).agg({
        'ValError': 'mean'
    }).round(4)
    
    summary.to_csv("results/procedural/phase4b_summary_table.csv")
    
    with open("results/procedural/phase4b_summary.md", "w") as f:
        f.write("# Phase 4B Results\n\n")
        f.write("## Overall Performance\n")
        
        flat_summary = summary.copy()
        flat_summary.columns = [
            'ValError_mean', 'ValError_std', 'SamplesWastedInactive', 
            'SamplesWastedNoisy', 'MasteredLearnable', 'FalseMastery', 'FalseBlocked'
        ]
        flat_summary['Val Error'] = flat_summary.apply(
            lambda row: f"{row['ValError_mean']:.4f} ± {row['ValError_std']:.4f}", axis=1
        )
        display_df = flat_summary.drop(columns=['ValError_mean', 'ValError_std'])
        cols = ['Val Error'] + [c for c in display_df.columns if c != 'Val Error']
        display_df = display_df[cols]
        
        f.write(display_df.to_markdown())
        f.write("\n\n")
        
        f.write("## Performance by Number of Active Regions\n")
        f.write(summary_by_region.to_markdown())
        
    print("Phase 4B Evaluation complete. Results saved in results/procedural/")

if __name__ == "__main__":
    evaluate_phase4b()

import os
import yaml
import copy
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from .world_generator import ProceduralEnvironment
from .agent import MasteryGatedHypothesisAgent
from .baselines import RandomAgent, HighestErrorAgent, LearningProgressAgent
from .evaluate_rl import PPOAgentWrapper

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_procedural():
    base_config = load_config()
    os.makedirs("results/procedural", exist_ok=True)
    
    num_regions = 10
    total_steps = base_config['experiment']['total_steps']
    num_seeds = 10
    
    # We want to test two env sets
    env_sets = {
        'ID_Medium': 'medium',
        'OOD_Hard': 'hard'
    }
    
    methods = [
        'MasteryGatedHypothesis',
        'Random', 'HighestError', 'LearningProgress'
    ]
    
    agents_classes = {
        'Random': RandomAgent,
        'HighestError': HighestErrorAgent,
        'LearningProgress': LearningProgressAgent,
        'MasteryGatedHypothesis': MasteryGatedHypothesisAgent
    }
    
    ppo_models = {
        'PPO_Fixed': 'results/rl/ppo_variant_b.zip',
        'PPO_Procedural': 'results/procedural/ppo_procedural.zip'
    }
    
    all_logs = []
    
    for env_name, difficulty in env_sets.items():
        print(f"--- Evaluating on {env_name} Worlds ---")
        
        # Hand-coded
        for method in methods:
            print(f"Evaluating {method}...")
            for seed in range(num_seeds):
                env = ProceduralEnvironment(difficulty=difficulty, max_regions=num_regions, seed=seed)
                agent = agents_classes[method](num_regions, base_config['agent'], seed)
                
                samples_wasted_inactive = 0
                samples_wasted_noisy = 0
                
                for _ in range(total_steps):
                    r = agent.select_action()
                    if env.active_mask[r] == 0.0:
                        samples_wasted_inactive += 1
                        # Wait, hand-coded agents should ideally not select inactive if they have a mechanism, 
                        # but they don't know about active_mask! They will just see y=0 always for padded regions.
                        # Actually, padded regions are constant 0 and have 0 noise. 
                        # They will just learn it perfectly immediately. 
                    else:
                        x = agent.select_x_for_region(r)
                        y = env.measure(r, x)
                        agent.record(r, x, y)
                        agent.train(env)
                        if not env.regions[r]['is_learnable']:
                            samples_wasted_noisy += 1
                            
                    # Note: to be perfectly fair, if they select inactive, we just return a measure of the dummy region.
                    # The dummy region is constant 0. We'll just run it properly.
                    if env.active_mask[r] == 0.0:
                        x = agent.select_x_for_region(r)
                        y = env.measure(r, x)
                        agent.record(r, x, y)
                        agent.train(env)
                        
                tracker = agent.tracker
                learnable_errs = []
                mastered_learnable = 0
                false_masteries = 0
                false_blocked = 0
                
                for r in range(env.num_active):
                    status = tracker.status[r]
                    err = float(tracker.val_error[r])
                    if env.regions[r]['is_learnable']:
                        learnable_errs.append(err)
                        if status == 'mastered':
                            mastered_learnable += 1
                        if status == 'blocked_noisy':
                            false_blocked += 1
                    else:
                        if status == 'mastered':
                            false_masteries += 1
                            
                all_logs.append({
                    'Method': method,
                    'Env Set': env_name,
                    'Seed': seed,
                    'Val Error (Learnable Avg)': np.mean(learnable_errs) if learnable_errs else 0.0,
                    'Samples Wasted Inactive': samples_wasted_inactive,
                    'Samples Wasted Noisy': samples_wasted_noisy,
                    'Total Mastered Learnable': mastered_learnable,
                    'False Masteries': false_masteries,
                    'False Blocked': false_blocked
                })
                
        # PPO
        for ppo_name, ppo_path in ppo_models.items():
            print(f"Evaluating {ppo_name}...")
            model = PPO.load(ppo_path)
            
            conf = copy.deepcopy(base_config)
            conf['environment']['procedural'] = True
            conf['environment']['difficulty'] = difficulty
            conf['environment']['max_regions'] = num_regions
            
            for seed in range(num_seeds):
                agent_wrapper = PPOAgentWrapper(model, num_regions, conf, seed)
                samples_wasted_inactive = 0
                samples_wasted_noisy = 0
                
                for _ in range(total_steps):
                    r, rew = agent_wrapper.step()
                    if agent_wrapper.env.env.active_mask[r] == 0.0:
                        samples_wasted_inactive += 1
                    else:
                        if not agent_wrapper.env.env.regions[r]['is_learnable']:
                            samples_wasted_noisy += 1
                            
                tracker = agent_wrapper.tracker
                env_internal = agent_wrapper.env.env
                
                learnable_errs = []
                mastered_learnable = 0
                false_masteries = 0
                false_blocked = 0
                
                for r in range(env_internal.num_active):
                    status = tracker.status[r]
                    err = float(tracker.val_error[r])
                    if env_internal.regions[r]['is_learnable']:
                        learnable_errs.append(err)
                        if status == 'mastered':
                            mastered_learnable += 1
                        if status == 'blocked_noisy':
                            false_blocked += 1
                    else:
                        if status == 'mastered':
                            false_masteries += 1
                            
                all_logs.append({
                    'Method': ppo_name,
                    'Env Set': env_name,
                    'Seed': seed,
                    'Val Error (Learnable Avg)': np.mean(learnable_errs) if learnable_errs else 0.0,
                    'Samples Wasted Inactive': samples_wasted_inactive,
                    'Samples Wasted Noisy': samples_wasted_noisy,
                    'Total Mastered Learnable': mastered_learnable,
                    'False Masteries': false_masteries,
                    'False Blocked': false_blocked
                })

    df = pd.DataFrame(all_logs)
    
    agg_funcs = {
        'Val Error (Learnable Avg)': ['mean', 'std'],
        'Samples Wasted Inactive': ['mean'],
        'Samples Wasted Noisy': ['mean'],
        'Total Mastered Learnable': ['mean'],
        'False Masteries': ['mean'],
        'False Blocked': ['mean']
    }
    
    summary = df.groupby(['Env Set', 'Method']).agg(agg_funcs)
    
    formatted_summary = pd.DataFrame()
    for col in agg_funcs.keys():
        if len(agg_funcs[col]) == 2:
            formatted_summary[col] = summary[(col, 'mean')].map('{:.4f}'.format) + " ± " + summary[(col, 'std')].map('{:.4f}'.format)
        else:
            formatted_summary[col] = summary[(col, 'mean')].map('{:.4f}'.format)
            
    formatted_summary.to_csv("results/procedural/procedural_summary_table.csv")
    
    with open("results/procedural/procedural_summary.md", "w") as f:
        f.write("# Procedural World Generation (Phase 4) Results\n\n")
        f.write(formatted_summary.to_markdown())
        
    print("Phase 4 Evaluation complete. Results saved in results/procedural/")

if __name__ == "__main__":
    evaluate_procedural()

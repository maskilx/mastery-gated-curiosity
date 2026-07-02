import os
import yaml
import copy
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from .environment import Environment
from .agent import MasteryGatedAgent, MasteryGatedHypothesisAgent
from .baselines import RandomAgent, HighestErrorAgent, UncertaintyOnlyAgent, LearningProgressAgent
from .evaluate_rl import PPOAgentWrapper

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_phase3b():
    base_config = load_config()
    os.makedirs("results/rl", exist_ok=True)
    
    seeds = base_config['experiment']['seeds']
    total_steps = base_config['experiment']['total_steps']
    num_regions = base_config['environment']['num_regions']
    
    methods = [
        'MasteryGatedHypothesis',
        'Random', 'HighestError', 'UncertaintyOnly', 
        'LearningProgress'
    ]
    
    agents_classes = {
        'Random': RandomAgent,
        'HighestError': HighestErrorAgent,
        'UncertaintyOnly': UncertaintyOnlyAgent,
        'LearningProgress': LearningProgressAgent,
        'MasteryGated': MasteryGatedAgent,
        'MasteryGatedHypothesis': MasteryGatedHypothesisAgent
    }
    
    ppo_variants = {
        'PPO_Variant_A': 'results/rl/ppo_variant_a.zip',
        'PPO_Variant_B': 'results/rl/ppo_variant_b.zip',
        'PPO_Variant_C': 'results/rl/ppo_variant_c.zip'
    }
    
    variants_configs = {
        'PPO_Variant_A': {
            'step_cost': 0.001, 'newly_mastered_bonus': 0.05, 'noisy_region_penalty': 0.02,
            'false_mastery_penalty': 0.2, 'false_blocked_penalty': 0.2, 'learnable_difficulty_bonus': 0.0
        },
        'PPO_Variant_B': {
            'step_cost': 0.0002, 'newly_mastered_bonus': 0.1, 'noisy_region_penalty': 0.02,
            'false_mastery_penalty': 0.2, 'false_blocked_penalty': 0.2, 'learnable_difficulty_bonus': 0.0
        },
        'PPO_Variant_C': {
            'step_cost': 0.0002, 'newly_mastered_bonus': 0.1, 'noisy_region_penalty': 0.02,
            'false_mastery_penalty': 0.2, 'false_blocked_penalty': 0.2, 'learnable_difficulty_bonus': 0.01
        }
    }
    
    all_logs = []
    
    for method in methods:
        print(f"Evaluating {method}...")
        for seed in range(seeds):
            env = Environment(num_regions, base_config['environment']['noise_learnable'], base_config['environment']['noise_unlearnable'], seed=seed)
            agent = agents_classes[method](num_regions, base_config['agent'], seed)
            samples = {r: 0 for r in range(num_regions)}
            time_to_mastery = {r: 1000 for r in range(num_regions)}
            
            for step in range(total_steps):
                r = agent.select_action()
                x = agent.select_x_for_region(r)
                y = env.measure(r, x)
                agent.record(r, x, y)
                agent.train(env)
                samples[r] += 1
                
                for reg in range(num_regions):
                    if agent.tracker.status[reg] == 'mastered' and time_to_mastery[reg] == 1000:
                        time_to_mastery[reg] = step + 1
                        
            tracker = agent.tracker
            
            log = {
                'Method': method,
                'Seed': seed,
                'Total Reward': 0.0
            }
            
            learnable_errs = []
            mastered_learnable = 0
            false_masteries = 0
            false_blocked = 0
            
            for r in range(num_regions):
                err = float(tracker.val_error[r])
                log[f'Reg {r} Val Error'] = err
                log[f'Reg {r} Samples'] = samples[r]
                log[f'Reg {r} Time to Mastery'] = time_to_mastery[r]
                
                status = tracker.status[r]
                if r != 3:
                    learnable_errs.append(err)
                    if status == 'mastered':
                        mastered_learnable += 1
                    if status == 'blocked_noisy':
                        false_blocked += 1
                else:
                    if status == 'mastered':
                        false_masteries += 1
                        
            log['Val Error (Learnable Avg)'] = np.mean(learnable_errs)
            log['Total Mastered Learnable'] = mastered_learnable
            log['False Masteries (Reg 3)'] = false_masteries
            log['False Blocked'] = false_blocked
            log['Reg 3 Samples %'] = samples[3] / total_steps * 100
            
            all_logs.append(log)
            
    for v_name, v_path in ppo_variants.items():
        print(f"Evaluating {v_name}...")
        model = PPO.load(v_path)
        conf = copy.deepcopy(base_config)
        conf['rl'] = variants_configs[v_name]
        
        for seed in range(seeds):
            agent_wrapper = PPOAgentWrapper(model, num_regions, conf, seed)
            samples = {r: 0 for r in range(num_regions)}
            time_to_mastery = {r: 1000 for r in range(num_regions)}
            total_reward = 0
            
            for step in range(total_steps):
                r, rew = agent_wrapper.step()
                samples[r] += 1
                total_reward += rew
                
                for reg in range(num_regions):
                    if agent_wrapper.tracker.status[reg] == 'mastered' and time_to_mastery[reg] == 1000:
                        time_to_mastery[reg] = step + 1
                        
            tracker = agent_wrapper.tracker
            log = {
                'Method': v_name,
                'Seed': seed,
                'Total Reward': total_reward
            }
            
            learnable_errs = []
            mastered_learnable = 0
            false_masteries = 0
            false_blocked = 0
            
            for r in range(num_regions):
                err = float(tracker.val_error[r])
                log[f'Reg {r} Val Error'] = err
                log[f'Reg {r} Samples'] = samples[r]
                log[f'Reg {r} Time to Mastery'] = time_to_mastery[r]
                
                status = tracker.status[r]
                if r != 3:
                    learnable_errs.append(err)
                    if status == 'mastered':
                        mastered_learnable += 1
                    if status == 'blocked_noisy':
                        false_blocked += 1
                else:
                    if status == 'mastered':
                        false_masteries += 1
                        
            log['Val Error (Learnable Avg)'] = np.mean(learnable_errs)
            log['Total Mastered Learnable'] = mastered_learnable
            log['False Masteries (Reg 3)'] = false_masteries
            log['False Blocked'] = false_blocked
            log['Reg 3 Samples %'] = samples[3] / total_steps * 100
            
            all_logs.append(log)
            
    df = pd.DataFrame(all_logs)
    
    agg_funcs = {
        'Val Error (Learnable Avg)': ['mean', 'std'],
        'Reg 2 Val Error': ['mean', 'std'],
        'Reg 2 Samples': ['mean', 'std'],
        'Reg 2 Time to Mastery': ['mean', 'std'],
        'Reg 3 Samples': ['mean', 'std'],
        'False Masteries (Reg 3)': ['mean'],
        'False Blocked': ['mean'],
        'Total Reward': ['mean', 'std']
    }
    
    summary = df.groupby('Method').agg(agg_funcs)
    
    formatted_summary = pd.DataFrame()
    for col in agg_funcs.keys():
        if len(agg_funcs[col]) == 2:
            formatted_summary[col] = summary[(col, 'mean')].map('{:.4f}'.format) + " ± " + summary[(col, 'std')].map('{:.4f}'.format)
        else:
            formatted_summary[col] = summary[(col, 'mean')].map('{:.4f}'.format)
            
    formatted_summary.to_csv("results/rl/phase3b_reward_ablation.csv")
    
    with open("results/rl/phase3b_reward_ablation.md", "w") as f:
        f.write("# Phase 3B: Reward Ablation Summary\n\n")
        f.write(formatted_summary.to_markdown())
        
    print("Phase 3B Evaluation complete. Results saved in results/rl/")

if __name__ == "__main__":
    evaluate_phase3b()

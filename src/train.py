import yaml
import os
import json
import numpy as np
from .environment import Environment
from .agent import MasteryGatedAgent, MasteryGatedHypothesisAgent
from .baselines import RandomAgent, HighestErrorAgent, UncertaintyOnlyAgent, LearningProgressAgent

import pandas as pd

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(method_name, seed, config):
    env = Environment(
        num_regions=config['environment']['num_regions'],
        noise_learnable=config['environment']['noise_learnable'],
        noise_unlearnable=config['environment']['noise_unlearnable'],
        seed=seed
    )
    
    agents = {
        'Random': RandomAgent,
        'HighestError': HighestErrorAgent,
        'UncertaintyOnly': UncertaintyOnlyAgent,
        'LearningProgress': LearningProgressAgent,
        'MasteryGated': MasteryGatedAgent,
        'MasteryGatedHypothesis': MasteryGatedHypothesisAgent
    }
    
    agent = agents[method_name](num_regions=config['environment']['num_regions'], config=config['agent'], seed=seed)
    total_steps = config['experiment']['total_steps']
    
    logs = {
        'method': method_name,
        'seed': seed,
        'selected_regions': [],
        'val_error_history': {r: [] for r in range(env.num_regions)},
        'lp_history': {r: [] for r in range(env.num_regions)},
        'status_history': {r: [] for r in range(env.num_regions)},
        'samples_per_region': {r: 0 for r in range(env.num_regions)},
        'false_masteries': 0,
        'false_blocked': 0
    }
    
    mastered_regions = set()
    blocked_regions = set()
    trajectory_data = []
    
    for step in range(total_steps):
        # Snapshot state before action
        state_features = {}
        for r_id in range(env.num_regions):
            state_features[r_id] = {
                'samples': agent.tracker.num_samples[r_id],
                'val_error': float(agent.tracker.val_error[r_id]),
                'uncertainty': float(agent.tracker.uncertainty[r_id]),
                'lp': float(agent.tracker.learning_progress[r_id]),
                'noise_risk': float(agent.tracker.noise_risk[r_id]),
                'status': agent.tracker.status[r_id]
            }
            
        r = agent.select_action()
        x = agent.select_x_for_region(r)
        
        val_error_before = state_features[r]['val_error']
        
        y = float(env.measure(r, x))
        
        agent.record(r, x, y)
        agent.train(env)
        
        val_error_after = float(agent.tracker.val_error[r])
        achieved_lp = val_error_before - val_error_after
        
        # Build trajectory row
        row = {
            'step': step,
            'seed': seed,
            'method': method_name,
            'selected_region': r,
            'selected_x': float(x),
            'observed_y': y,
            'val_error_before': val_error_before,
            'val_error_after': val_error_after,
            'achieved_lp': achieved_lp
        }
        
        # Add per-region features to row
        for r_id in range(env.num_regions):
            for k, v in state_features[r_id].items():
                row[f'region_{r_id}_{k}'] = v
                
        trajectory_data.append(row)
        
        logs['selected_regions'].append(int(r))
        logs['samples_per_region'][r] += 1
        
        for region_id in range(env.num_regions):
            logs['val_error_history'][region_id].append(float(agent.tracker.val_error[region_id]))
            logs['lp_history'][region_id].append(float(agent.tracker.learning_progress[region_id]))
            
            status = agent.tracker.status[region_id]
            logs['status_history'][region_id].append(status)
            
            if status == 'mastered' and region_id not in mastered_regions:
                mastered_regions.add(region_id)
                if region_id == 3:
                    logs['false_masteries'] += 1
                    
            if status == 'blocked_noisy' and region_id not in blocked_regions:
                blocked_regions.add(region_id)
                if region_id != 3:
                    logs['false_blocked'] += 1
                    
    # Post-process trajectory for later mastered/blocked
    final_status = {r: agent.tracker.status[r] for r in range(env.num_regions)}
    for row in trajectory_data:
        r_selected = row['selected_region']
        row['later_mastered'] = (final_status[r_selected] == 'mastered')
        row['later_blocked_noisy'] = (final_status[r_selected] == 'blocked_noisy')
        row['final_status_of_selected'] = final_status[r_selected]
        
    df = pd.DataFrame(trajectory_data)
    os.makedirs("results/trajectories", exist_ok=True)
    df.to_csv(f"results/trajectories/trajectory_seed_{seed}_{method_name}.csv", index=False)
        
    return logs

if __name__ == "__main__":
    os.makedirs("results/logs", exist_ok=True)
    config = load_config()
    seeds = config['experiment']['seeds']
    methods = ['Random', 'HighestError', 'UncertaintyOnly', 'LearningProgress', 'MasteryGated', 'MasteryGatedHypothesis']
    
    all_logs = []
    
    for method in methods:
        print(f"Running {method}...")
        for seed in range(seeds):
            print(f"  Seed {seed}...")
            logs = run_experiment(method, seed, config)
            all_logs.append(logs)
            
    with open("results/logs/all_logs.json", "w") as f:
        json.dump(all_logs, f)
    print("Training complete. Logs saved.")

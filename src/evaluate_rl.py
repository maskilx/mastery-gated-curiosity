import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from .environment import Environment
from .agent import MasteryGatedAgent, MasteryGatedHypothesisAgent
from .baselines import RandomAgent, HighestErrorAgent, UncertaintyOnlyAgent, LearningProgressAgent
from .rl_env import CuriosityEnv

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class PPOAgentWrapper:
    def __init__(self, model, num_regions, config, seed):
        self.model = model
        self.env = CuriosityEnv(config, seed=seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.tracker = self.env.agent.tracker
        
    def step(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        return int(action), reward

def evaluate_rl():
    config = load_config()
    os.makedirs("results/rl", exist_ok=True)
    
    seeds = config['experiment']['seeds']
    total_steps = config['experiment']['total_steps']
    num_regions = config['environment']['num_regions']
    
    methods = [
        'Random', 'HighestError', 'UncertaintyOnly', 
        'LearningProgress', 'MasteryGated', 'MasteryGatedHypothesis', 
        'PPOCuriosityRegion'
    ]
    
    agents_classes = {
        'Random': RandomAgent,
        'HighestError': HighestErrorAgent,
        'UncertaintyOnly': UncertaintyOnlyAgent,
        'LearningProgress': LearningProgressAgent,
        'MasteryGated': MasteryGatedAgent,
        'MasteryGatedHypothesis': MasteryGatedHypothesisAgent
    }
    
    ppo_model = PPO.load("results/rl/ppo_curiosity_region.zip")
    
    all_logs = []
    
    for method in methods:
        print(f"Evaluating {method}...")
        for seed in range(seeds):
            if method == 'PPOCuriosityRegion':
                agent_wrapper = PPOAgentWrapper(ppo_model, num_regions, config, seed)
                samples = {r: 0 for r in range(num_regions)}
                total_reward = 0
                for _ in range(total_steps):
                    r, rew = agent_wrapper.step()
                    samples[r] += 1
                    total_reward += rew
                tracker = agent_wrapper.tracker
            else:
                env = Environment(num_regions, config['environment']['noise_learnable'], config['environment']['noise_unlearnable'], seed=seed)
                agent = agents_classes[method](num_regions, config['agent'], seed)
                samples = {r: 0 for r in range(num_regions)}
                total_reward = 0
                for _ in range(total_steps):
                    r = agent.select_action()
                    x = agent.select_x_for_region(r)
                    y = env.measure(r, x)
                    agent.record(r, x, y)
                    agent.train(env)
                    samples[r] += 1
                tracker = agent.tracker
                
            # Log metrics
            log = {
                'Method': method,
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
    
    # Aggregate
    agg_funcs = {
        'Val Error (Learnable Avg)': ['mean', 'std'],
        'Reg 2 Val Error': ['mean', 'std'],
        'Reg 3 Samples': ['mean', 'std'],
        'Reg 3 Samples %': ['mean', 'std'],
        'Total Mastered Learnable': ['mean', 'std'],
        'False Masteries (Reg 3)': ['mean'],
        'False Blocked': ['mean'],
        'Total Reward': ['mean', 'std']
    }
    
    summary = df.groupby('Method').agg(agg_funcs)
    
    # Format
    formatted_summary = pd.DataFrame()
    for col in agg_funcs.keys():
        if len(agg_funcs[col]) == 2:
            formatted_summary[col] = summary[(col, 'mean')].map('{:.4f}'.format) + " ± " + summary[(col, 'std')].map('{:.4f}'.format)
        else:
            formatted_summary[col] = summary[(col, 'mean')].map('{:.4f}'.format)
            
    formatted_summary.to_csv("results/rl/rl_summary_table.csv")
    
    # Save markdown
    with open("results/rl/rl_summary.md", "w") as f:
        f.write("# RL Evaluation Summary\n\n")
        f.write(formatted_summary.to_markdown())
        
    print("Evaluation complete. Results saved in results/rl/")
    
if __name__ == "__main__":
    evaluate_rl()

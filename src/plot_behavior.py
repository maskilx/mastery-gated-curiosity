import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from src.rl_env import CuriosityEnv

def plot_episode_behavior(agent, ppo_env, env_type, seed=42):
    obs, _ = ppo_env.reset(seed=seed)
    
    selected_regions = []
    val_errors = {r: [] for r in range(10)}
    samples = {r: [] for r in range(10)}
    statuses = {r: [] for r in range(10)}
    
    noisy_mask = np.zeros(10)
    for r in range(10):
        if ppo_env.env.active_mask[r] == 1.0 and not ppo_env.env.regions[r]['is_learnable']:
            noisy_mask[r] = 1.0
            
    noisy_selections = []
    
    done = False
    step_count = 0
    while not done:
        mask = ppo_env.action_masks()
        action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
        obs, _, done, _, _ = ppo_env.step(action)
        
        region = int(action)
        selected_regions.append(region)
        
        if noisy_mask[region] == 1.0:
            noisy_selections.append(1)
        else:
            noisy_selections.append(0)
            
        tracker = ppo_env.agent.tracker
        for r in range(10):
            if ppo_env.env.active_mask[r] == 1.0:
                val_errors[r].append(tracker.val_error[r])
                samples[r].append(tracker.num_samples[r])
                statuses[r].append(tracker.status[r])
            else:
                val_errors[r].append(np.nan)
                samples[r].append(np.nan)
                statuses[r].append("inactive")
                
        step_count += 1
        if step_count >= 1000:
            done = True
            
    # Plotting
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # 1. Selected Region over Time
    axs[0].scatter(range(len(selected_regions)), selected_regions, alpha=0.5, s=10)
    axs[0].set_ylabel('Selected Region')
    axs[0].set_title(f'PPO_SetPolicy Behavior - {env_type} (Seed {seed})')
    
    # 2. Status Timeline
    status_map = {'unexplored': 0, 'learning': 1, 'mastered': 2, 'blocked_noisy': 3, 'inactive': -1}
    for r in range(10):
        if ppo_env.env.active_mask[r] == 1.0:
            s_vals = [status_map[s] for s in statuses[r]]
            axs[1].plot(s_vals, label=f'R{r}')
    axs[1].set_ylabel('Status')
    axs[1].set_yticks([0, 1, 2, 3])
    axs[1].set_yticklabels(['Unexplored', 'Learning', 'Mastered', 'Blocked'])
    axs[1].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # 3. Validation Error
    for r in range(10):
        if ppo_env.env.active_mask[r] == 1.0:
            axs[2].plot(val_errors[r], label=f'R{r}')
    axs[2].set_ylabel('Validation Error')
    axs[2].set_yscale('log')
    
    # 4. Cumulative Samples
    for r in range(10):
        if ppo_env.env.active_mask[r] == 1.0:
            axs[3].plot(samples[r], label=f'R{r}')
    axs[3].set_ylabel('Cumulative Samples')
    
    # 5. Noisy Region Selections (Cumulative)
    axs[4].plot(np.cumsum(noisy_selections), color='red')
    axs[4].set_ylabel('Cumulative Noisy Samples')
    axs[4].set_xlabel('Step')
    
    plt.tight_layout()
    plt.savefig(f'results/procedural/behavior_{env_type}.png')
    plt.close()

def plot_behavior():
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['environment']['max_regions'] = 10
    config['environment']['procedural'] = True
    
    # ID Medium
    config['environment']['difficulty'] = 'medium'
    ppo_env = CuriosityEnv(config, seed=42)
    agent_ppo_set = MaskablePPO.load('results/procedural/ppo_set_policy', env=ppo_env)
    plot_episode_behavior(agent_ppo_set, ppo_env, 'ID_Medium', seed=42)
    
    # OOD Hard
    config['environment']['difficulty'] = 'hard'
    ppo_env = CuriosityEnv(config, seed=123)
    agent_ppo_set = MaskablePPO.load('results/procedural/ppo_set_policy', env=ppo_env)
    plot_episode_behavior(agent_ppo_set, ppo_env, 'OOD_Hard', seed=123)

if __name__ == '__main__':
    plot_behavior()

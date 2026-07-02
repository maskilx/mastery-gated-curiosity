import os
import yaml
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from src.rl_env import CuriosityEnv, CuriosityEnvNoValObs, CuriosityEnvNoValObs_NoStatus
from src.agent import MasteryGatedHypothesisAgent
from src.baselines import OracleBestRegionAgent

def evaluate_agent(agent_type, ppo_path=None, ppo_env=None, use_novalobs=False, 
                   budget=1000, forced_families=None, difficulty='medium', num_seeds=5):
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['environment']['procedural'] = True
    config['environment']['difficulty'] = difficulty
    config['environment']['max_regions'] = 10
    config['environment']['fixed_num_active'] = None
    config['environment']['forced_families'] = forced_families
    config['experiment']['total_steps'] = budget
    
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=num_seeds)
    
    results = []
    
    for seed in seeds:
        seed = int(seed)
        if agent_type.startswith('PPO'):
            ppo_env.config = config # Update config to inject budget and forced_families
            ppo_env.max_steps = budget
            ppo_env.reset(seed=seed)
            env = ppo_env.env
        else:
            if agent_type == 'OracleBestRegion':
                agent = OracleBestRegionAgent(10, config['agent'], seed=seed)
            elif agent_type == 'MasteryGatedHypothesis':
                agent = MasteryGatedHypothesisAgent(10, config['agent'], seed=seed)
            elif agent_type == 'RandomMask':
                from src.agent import BaseAgent
                agent = BaseAgent(10, config['agent'], seed=seed)
                agent.select_action = lambda active_mask=None: np.random.choice(np.where(active_mask == 1.0)[0])
                agent.select_x_for_region = lambda r: np.random.uniform(-1, 1)
            elif agent_type == 'LearningProgress':
                from src.baselines import LearningProgressAgent
                agent = LearningProgressAgent(10, config['agent'], seed=seed)
                hyp = MasteryGatedHypothesisAgent(10, config['agent'], seed=seed)
                agent.select_x_for_region = lambda r: hyp.select_x_for_region(r)
                
            from src.world_generator import ProceduralEnvironment
            env = ProceduralEnvironment(
                difficulty=difficulty,
                max_regions=10,
                seed=seed,
                forced_families=forced_families
            )
            
        if agent_type.startswith('PPO'):
            ppo_env.step_count = 0
            agent = MaskablePPO.load(ppo_path, env=ppo_env)
        else:
            step_count = 0
            
        done = False
        
        while not done:
            if agent_type.startswith('PPO'):
                obs = ppo_env._get_obs()
                mask = ppo_env.action_masks()
                action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
                _, _, done, _, _ = ppo_env.step(action)
            else:
                action = agent.select_action(active_mask=env.active_mask)
                x = agent.select_x_for_region(action)
                y = env.measure(action, x)
                agent.record(action, x, y)
                agent.train(env)
                step_count += 1
                if step_count >= budget:
                    done = True
                    
        tracker = ppo_env.agent.tracker if agent_type.startswith('PPO') else agent.tracker
        actual_model = ppo_env.agent.model if agent_type.startswith('PPO') else agent.model
        
        active_indices = np.where(env.active_mask == 1.0)[0]
        learnable_indices = [r for r in active_indices if env.regions[r]['family'] != 'noisy_unlearnable']
        noisy_indices = [r for r in active_indices if env.regions[r]['family'] == 'noisy_unlearnable']
        inactive_indices = np.where(env.active_mask == 0.0)[0]
        
        true_val_errors = {}
        for r in active_indices:
            val_x, val_y = env.get_validation_set(r)
            vec = np.zeros(10)
            vec[r] = 1.0
            val_X = np.array([np.concatenate([vec, [vx]]) for vx in val_x])
            mean_pred, _ = actual_model.predict_with_uncertainty(val_X)
            true_val_errors[r] = np.mean((val_y - mean_pred)**2)
            
            # Log per-family breakdown
            results.append({
                'Agent': agent_type,
                'Budget': budget,
                'Condition': forced_families[0] if forced_families and len(forced_families)==1 else 'Mixed',
                'Seed': seed,
                'Region': r,
                'Family': env.regions[r]['family'],
                'IsActive': True,
                'IsNoisy': env.regions[r]['family'] == 'noisy_unlearnable',
                'ValError': true_val_errors[r],
                'NumSamples': tracker.num_samples[r],
                'Status': tracker.status[r]
            })
            
    return results

def evaluate_phase4d():
    os.makedirs('results/procedural', exist_ok=True)
    
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['environment']['max_regions'] = 10
    config['environment']['procedural'] = True
    
    env_standard = CuriosityEnv(config, seed=42)
    env_novalobs = CuriosityEnvNoValObs(config, seed=42)
    env_nostatus = CuriosityEnvNoValObs_NoStatus(config, seed=42)
    
    agents_to_eval = [
        ('RandomMask', None, None, False),
        ('LearningProgress', None, None, False),
        ('MasteryGatedHypothesis', None, None, False),
        ('OracleBestRegion', None, None, False),
        ('PPO_SetPolicy_Phase4B', 'results/procedural/ppo_set_policy.zip', env_standard, False),
        ('PPO_SetPolicy_NoValObs', 'results/procedural/ppo_set_policy_novalobs.zip', env_novalobs, True),
        ('PPO_SetPolicy_NoValObs_NoStatus', 'results/procedural/ppo_set_policy_novalobs_nostatus.zip', env_nostatus, True)
    ]
    
    all_results = []
    
    # 1. Budget Sweep
    print("--- Running Budget Sweep ---")
    budgets = [50, 100, 200, 500, 1000]
    for budget in budgets:
        print(f"Budget: {budget}")
        for name, ppo_path, ppo_env, noval in agents_to_eval:
            try:
                res = evaluate_agent(name, ppo_path=ppo_path, ppo_env=ppo_env, use_novalobs=noval, budget=budget, difficulty='medium', num_seeds=5)
                all_results.extend(res)
            except Exception as e:
                print(f"Skipping {name} (may not be trained yet): {e}")

    # 2. Family-Heavy OOD Sweep & Deceptive Sweep (Budget = 1000)
    print("--- Running Family-Heavy OOD & Deceptive Sweep ---")
    families_to_test = [
        ['high_freq_sine'],
        ['exp'],
        ['log'],
        ['piecewise_linear'],
        ['cubic'],
        ['deceptive_noisy_to_clean'],
        ['deceptive_simple_to_complex'],
        ['deceptive_narrow_peak'],
        ['noisy_learnable']
    ]
    
    for f in families_to_test:
        print(f"Family-Heavy: {f[0]}")
        for name, ppo_path, ppo_env, noval in agents_to_eval:
            try:
                res = evaluate_agent(name, ppo_path=ppo_path, ppo_env=ppo_env, use_novalobs=noval, budget=1000, forced_families=f, difficulty='stress_test', num_seeds=5)
                all_results.extend(res)
            except Exception as e:
                print(f"Skipping {name} (may not be trained yet): {e}")

    df = pd.DataFrame(all_results)
    df.to_csv('results/procedural/phase4d_stress_table.csv', index=False)
    print("Saved raw data to results/procedural/phase4d_stress_table.csv")
    
    # Generate Summaries
    
    # Budget Summary
    df_learnable = df[~df['IsNoisy']]
    df_noisy = df[df['IsNoisy']]
    
    budget_summary = df_learnable[df_learnable['Condition'] == 'Mixed'].groupby(['Budget', 'Agent'])['ValError'].mean().unstack()
    budget_noisy_samples = df_noisy[df_noisy['Condition'] == 'Mixed'].groupby(['Budget', 'Agent'])['NumSamples'].mean().unstack()
    
    # Family Summary (Budget = 1000)
    family_summary = df_learnable[(df_learnable['Budget'] == 1000) & (df_learnable['Condition'] != 'Mixed')].groupby(['Family', 'Agent'])['ValError'].mean().unstack()
    
    with open('results/procedural/phase4d_stress_summary.md', 'w') as f:
        f.write("# Phase 4D: Stress Test Summary\n\n")
        f.write("## Budget Sweep (Learnable Val Error)\n\n")
        f.write(budget_summary.to_markdown())
        f.write("\n\n## Budget Sweep (Noisy Samples Wasted)\n\n")
        if not budget_noisy_samples.empty:
            f.write(budget_noisy_samples.to_markdown())
        else:
            f.write("No noisy regions tested in this split.\n")
        f.write("\n\n## Family-Heavy OOD & Deceptive Families (Val Error)\n\n")
        f.write(family_summary.to_markdown())
        f.write("\n")
        
    # Plotting
    import matplotlib.pyplot as plt
    
    # 1. Budget Sweep Plot
    plt.figure(figsize=(10, 6))
    for agent in budget_summary.columns:
        plt.plot(budget_summary.index, budget_summary[agent], marker='o', label=agent)
    plt.xscale('log')
    plt.xlabel('Step Budget')
    plt.ylabel('Mean Validation Error (Learnable Regions)')
    plt.title('Performance vs. Step Budget')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/procedural/phase4d_budget_sweep.png')
    plt.close()
    
    # 2. Per-Family Breakdown Plot
    plt.figure(figsize=(12, 6))
    family_summary.plot(kind='bar', figsize=(14, 6))
    plt.ylabel('Mean Validation Error')
    plt.title('Performance by OOD / Deceptive Family (Budget = 1000)')
    plt.legend(title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/procedural/phase4d_family_breakdown.png')
    plt.close()
    
    print("Phase 4D evaluation complete.")

if __name__ == '__main__':
    evaluate_phase4d()

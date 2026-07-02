import os
import yaml
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from src.rl_env import CuriosityEnv
from src.baselines import OracleBestRegionAgent, OracleRandomLearnableAgent

def evaluate_agent(agent, agent_type, env_type, ppo_env=None, ablate_random_x=False):
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['environment']['procedural'] = True
    config['environment']['difficulty'] = env_type.split('_')[1].lower()
    
    # We want max regions = 10, variable active
    config['environment']['max_regions'] = 10
    config['environment']['fixed_num_active'] = None
    config['experiment']['total_steps'] = 1000
    
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=10)
    
    results = []
    
    for seed in seeds:
        seed = int(seed)
        if agent_type.startswith('PPO'):
            ppo_env.reset(seed=seed)
            env = ppo_env.env
        else:
            if agent_type == 'OracleBestRegion':
                agent = OracleBestRegionAgent(10, config['agent'], seed=seed)
            elif agent_type == 'OracleRandomLearnable':
                agent = OracleRandomLearnableAgent(10, config['agent'], seed=seed)
                
            from src.world_generator import ProceduralEnvironment
            env = ProceduralEnvironment(
                difficulty=config['environment']['difficulty'],
                max_regions=10,
                seed=seed
            )
            
        if agent_type.startswith('PPO'):
            ppo_env.step_count = 0
            if ablate_random_x:
                # Monkey-patch to force random X instead of HypothesisDiscriminator
                ppo_env.agent.select_x_for_region = lambda r: np.random.uniform(-1, 1)
                
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
                action = agent.select_action(active_mask=env.active_mask, env=env)
                x = agent.select_x_for_region(action)
                y = env.measure(action, x)
                agent.record(action, x, y)
                agent.train(env)
                step_count += 1
                if step_count >= config['experiment']['total_steps']:
                    done = True
                    
        tracker = ppo_env.agent.tracker if agent_type.startswith('PPO') else agent.tracker
        
        active_indices = np.where(env.active_mask == 1.0)[0]
        learnable_indices = [r for r in active_indices if env.regions[r]['family'] != 'noisy_unlearnable']
        noisy_indices = [r for r in active_indices if env.regions[r]['family'] == 'noisy_unlearnable']
        inactive_indices = np.where(env.active_mask == 0.0)[0]
        
        val_error = 0.0
        if len(learnable_indices) > 0:
            val_error = np.mean([tracker.val_error[r] for r in learnable_indices])
            
        wasted_noisy = sum([tracker.num_samples[r] for r in noisy_indices])
        wasted_inactive = sum([tracker.num_samples[r] for r in inactive_indices])
        
        mastered_learnable = sum([1 for r in learnable_indices if tracker.status[r] == 'mastered'])
        false_mastery = sum([1 for r in noisy_indices if tracker.status[r] == 'mastered'])
        false_blocked = sum([1 for r in learnable_indices if tracker.status[r] == 'blocked_noisy'])
        
        results.append({
            'EnvType': env_type,
            'Agent': agent_type + ('_RandomX' if ablate_random_x else ''),
            'NumActive': len(active_indices),
            'Seed': seed,
            'ValError': val_error,
            'SamplesWastedInactive': wasted_inactive,
            'SamplesWastedNoisy': wasted_noisy,
            'MasteredLearnable': mastered_learnable,
            'FalseMastery': false_mastery,
            'FalseBlocked': false_blocked
        })
        
    return results

def evaluate_audit():
    os.makedirs('results/procedural', exist_ok=True)
    
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['environment']['max_regions'] = 10
    config['environment']['procedural'] = True
    ppo_env = CuriosityEnv(config, seed=42)
    
    agent_ppo_set = MaskablePPO.load('results/procedural/ppo_set_policy', env=ppo_env)
    
    agents_to_eval = [
        ('OracleBestRegion', False),
        ('OracleRandomLearnable', False),
        ('PPO_SetPolicy', False),
        ('PPO_SetPolicy', True) # RandomX ablation
    ]
    
    envs_to_eval = ['ID_Medium', 'OOD_Hard']
    
    all_results = []
    
    for env_type in envs_to_eval:
        print(f"--- Evaluating on {env_type} ---")
        for agent_name, ablate in agents_to_eval:
            print(f"Evaluating {agent_name} (Ablated: {ablate})...")
            if agent_name.startswith('PPO'):
                res = evaluate_agent(agent_ppo_set, agent_name, env_type, ppo_env=ppo_env, ablate_random_x=ablate)
                agent_ppo_set = MaskablePPO.load('results/procedural/ppo_set_policy', env=ppo_env) # reload just in case
            else:
                res = evaluate_agent(None, agent_name, env_type)
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv('results/procedural/phase4b_audit_tables.csv', index=False)
    
    summary = df.groupby(['EnvType', 'Agent']).agg({
        'ValError': ['mean', 'std'],
        'SamplesWastedInactive': 'mean',
        'SamplesWastedNoisy': 'mean',
        'MasteredLearnable': 'mean',
        'FalseMastery': 'mean',
        'FalseBlocked': 'mean'
    }).round(4)
    
    with open('results/procedural/phase4b_audit.md', 'a') as f:
        f.write("\n## 3. Evaluation Audit & Oracle Baselines\n")
        f.write("Ablation of X-selection and evaluation against Oracle baselines.\n\n")
        f.write(summary.to_markdown())
        f.write("\n")
        
    print("Audit evaluation complete.")

if __name__ == '__main__':
    evaluate_audit()

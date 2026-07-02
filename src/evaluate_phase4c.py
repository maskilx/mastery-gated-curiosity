import os
import yaml
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from src.rl_env import CuriosityEnv, CuriosityEnvNoValObs
from src.agent import MasteryGatedHypothesisAgent
from src.baselines import OracleBestRegionAgent

def evaluate_agent(agent_type, env_type, ppo_path=None, ppo_env=None, use_novalobs=False):
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['environment']['procedural'] = True
    config['environment']['difficulty'] = env_type.split('_')[1].lower()
    
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
            elif agent_type == 'MasteryGatedHypothesis':
                agent = MasteryGatedHypothesisAgent(10, config['agent'], seed=seed)
            elif agent_type == 'RandomMask':
                from src.agent import BaseAgent
                agent = BaseAgent(10, config['agent'], seed=seed)
                agent.select_action = lambda active_mask=None: np.random.choice(np.where(active_mask == 1.0)[0])
                agent.select_x_for_region = lambda r: np.random.uniform(-1, 1)
                
            from src.world_generator import ProceduralEnvironment
            env = ProceduralEnvironment(
                difficulty=config['environment']['difficulty'],
                max_regions=10,
                seed=seed
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
                if step_count >= config['experiment']['total_steps']:
                    done = True
                    
        # For NoValObs env, the agent's tracker won't automatically have the TRUE validation error 
        # populated during training because we skipped querying it. 
        # However, we must compute it NOW for fair final evaluation!
        tracker = ppo_env.agent.tracker if agent_type.startswith('PPO') else agent.tracker
        actual_model = ppo_env.agent.model if agent_type.startswith('PPO') else agent.model
        
        active_indices = np.where(env.active_mask == 1.0)[0]
        learnable_indices = [r for r in active_indices if env.regions[r]['family'] != 'noisy_unlearnable']
        noisy_indices = [r for r in active_indices if env.regions[r]['family'] == 'noisy_unlearnable']
        inactive_indices = np.where(env.active_mask == 0.0)[0]
        
        # Manually compute final generalization error across the true hidden sets
        true_val_errors = {}
        for r in active_indices:
            val_x, val_y = env.get_validation_set(r)
            # Create one hot representation for inference
            vec = np.zeros(10)
            vec[r] = 1.0
            val_X = np.array([np.concatenate([vec, [vx]]) for vx in val_x])
            mean_pred, _ = actual_model.predict_with_uncertainty(val_X)
            true_val_errors[r] = np.mean((val_y - mean_pred)**2)
            
        val_error = 0.0
        if len(learnable_indices) > 0:
            val_error = np.mean([true_val_errors[r] for r in learnable_indices])
            
        wasted_noisy = sum([tracker.num_samples[r] for r in noisy_indices])
        wasted_inactive = sum([tracker.num_samples[r] for r in inactive_indices])
        
        mastered_learnable = sum([1 for r in learnable_indices if tracker.status[r] == 'mastered'])
        false_mastery = sum([1 for r in noisy_indices if tracker.status[r] == 'mastered'])
        false_blocked = sum([1 for r in learnable_indices if tracker.status[r] == 'blocked_noisy'])
        
        results.append({
            'EnvType': env_type,
            'Agent': agent_type,
            'NumActive': len(active_indices),
            'Seed': seed,
            'ValError': val_error,
            'SamplesWastedInactive': wasted_inactive,
            'SamplesWastedNoisy': wasted_noisy,
            'MasteredLearnable': mastered_learnable,
            'FalseMastery': false_mastery,
            'FalseBlocked': false_blocked,
            'UsedHiddenValidation': not use_novalobs if agent_type.startswith('PPO') else True
        })
        
    return results

def evaluate_phase4c():
    os.makedirs('results/procedural', exist_ok=True)
    
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['environment']['max_regions'] = 10
    config['environment']['procedural'] = True
    
    env_standard = CuriosityEnv(config, seed=42)
    env_novalobs = CuriosityEnvNoValObs(config, seed=42)
    
    agents_to_eval = [
        ('RandomMask', None, None, False),
        ('MasteryGatedHypothesis', None, None, False),
        ('OracleBestRegion', None, None, False),
        ('PPO_SetPolicy_Phase4B', 'results/procedural/ppo_set_policy.zip', env_standard, False),
        ('PPO_SetPolicy_NoValObs', 'results/procedural/ppo_set_policy_novalobs.zip', env_novalobs, True),
        ('PPO_FlatMLP_NoValObs', 'results/procedural/ppo_flat_policy_novalobs.zip', env_novalobs, True)
    ]
    
    envs_to_eval = ['ID_Medium', 'OOD_Hard']
    
    all_results = []
    
    for env_type in envs_to_eval:
        print(f"--- Evaluating on {env_type} ---")
        for name, ppo_path, ppo_env, noval in agents_to_eval:
            print(f"Evaluating {name}...")
            res = evaluate_agent(name, env_type, ppo_path=ppo_path, ppo_env=ppo_env, use_novalobs=noval)
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv('results/procedural/phase4c_summary_table.csv', index=False)
    
    summary = df.groupby(['EnvType', 'Agent']).agg({
        'ValError': ['mean', 'std'],
        'SamplesWastedInactive': 'mean',
        'SamplesWastedNoisy': 'mean',
        'MasteredLearnable': 'mean',
        'FalseMastery': 'mean',
        'FalseBlocked': 'mean',
        'UsedHiddenValidation': 'first'
    }).round(4)
    
    with open('results/procedural/phase4c_summary.md', 'w') as f:
        f.write("# Phase 4C: No Hidden Validation Signal Summary\n\n")
        f.write("Comparison of the RL policies trained with privileged validation signals vs observable prequential error signals.\n\n")
        f.write(summary.to_markdown())
        f.write("\n")
        
    print("Phase 4C evaluation complete.")

if __name__ == '__main__':
    evaluate_phase4c()

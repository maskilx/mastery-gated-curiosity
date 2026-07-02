import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.drone_2d_env import Custom2DDroneEnv

def is_stable(states):
    # Stable if:
    # abs(x) < 0.2
    # abs(z - 1.0) < 0.2
    # abs(theta) < 0.15
    # for 20 consecutive steps
    stable_count = 0
    first_stable_step = -1
    
    for i, s in enumerate(states):
        x, z, vx, vz, theta, theta_dot = s
        if abs(x) < 0.2 and abs(z - 1.0) < 0.2 and abs(theta) < 0.15:
            stable_count += 1
            if stable_count == 20 and first_stable_step == -1:
                first_stable_step = i - 19
        else:
            stable_count = 0
            
    return first_stable_step

def evaluate_agent(model_path, num_episodes=50, perturb=False):
    model = PPO.load(model_path)
    env = Custom2DDroneEnv(mode='hover', max_steps=500)
    
    results = {
        'crash_rate': 0.0,
        'final_alt_error': [],
        'final_ang_error': [],
        'energy_cost': [],
        'stable_episodes': 0
    }
    
    crashes = 0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        if perturb:
            # Overwrite initial state to be a difficult recovery state
            env.state = np.array([
                np.random.uniform(-1.0, 1.0), # x
                np.random.uniform(0.5, 2.0),  # z
                np.random.uniform(-5.0, 5.0), # vx
                np.random.uniform(-5.0, 5.0), # vz
                np.random.uniform(-1.0, 1.0), # theta
                np.random.uniform(-2.0, 2.0)  # theta_dot
            ], dtype=np.float32)
            obs = env.state
            
        states = [obs]
        actions = []
        
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            states.append(obs)
            actions.append(action)
            
            if term or trunc:
                if info.get('crashed', False):
                    crashes += 1
                break
                
        states = np.array(states)
        actions = np.array(actions)
        
        # Metrics
        results['energy_cost'].append(np.sum(actions**2))
        
        if not info.get('crashed', False):
            # Final 50 steps average error
            final_states = states[-50:]
            alt_err = np.mean(np.abs(final_states[:, 1] - 1.0))
            ang_err = np.mean(np.abs(final_states[:, 4]))
            results['final_alt_error'].append(alt_err)
            results['final_ang_error'].append(ang_err)
            
            if is_stable(states) != -1:
                results['stable_episodes'] += 1
                
    results['crash_rate'] = crashes / num_episodes
    results['final_alt_error'] = np.mean(results['final_alt_error']) if results['final_alt_error'] else float('inf')
    results['final_ang_error'] = np.mean(results['final_ang_error']) if results['final_ang_error'] else float('inf')
    results['energy_cost'] = np.mean(results['energy_cost'])
    results['stability_rate'] = results['stable_episodes'] / num_episodes
    
    return results

def plot_learning_curves():
    agents = ['scratch', 'survival', 'curiosity']
    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        log_path = f'results/drone/logs/{agent}/evaluations.npz'
        if os.path.exists(log_path):
            data = np.load(log_path)
            timesteps = data['timesteps']
            rewards = np.mean(data['results'], axis=1)
            plt.plot(timesteps, rewards, label=agent)
            
    plt.xlabel('Timesteps')
    plt.ylabel('Eval Reward')
    plt.title('Hover Downstream Learning Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/drone/learning_curves.png')
    plt.close()

def run_evaluation():
    models = {
        'PPO_Scratch': 'results/drone/models/scratch_best/best_model.zip',
        'PPO_SurvivalPretrain': 'results/drone/models/survival_best/best_model.zip',
        'PPO_CuriosityPretrain': 'results/drone/models/curiosity_best/best_model.zip'
    }
    
    records = []
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, model not found.")
            continue
            
        print(f"Evaluating {name} (Standard)...")
        std_res = evaluate_agent(path, perturb=False)
        
        print(f"Evaluating {name} (Perturbed)...")
        pert_res = evaluate_agent(path, perturb=True)
        
        records.append({
            'Agent': name,
            'CrashRate (Std)': std_res['crash_rate'],
            'Stability (Std)': std_res['stability_rate'],
            'Alt Error': std_res['final_alt_error'],
            'Angle Error': std_res['final_ang_error'],
            'Energy': std_res['energy_cost'],
            'CrashRate (Perturb)': pert_res['crash_rate'],
            'Stability (Perturb)': pert_res['stability_rate'],
        })
        
    df = pd.DataFrame(records)
    df.to_csv('results/drone/phase5_results.csv', index=False)
    
    with open('results/drone/phase5_summary.md', 'w') as f:
        f.write("# Phase 5B: Curiosity Drone Control\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n![Learning Curves](/Users/adimaskil/.gemini/antigravity/brain/1ca23ce5-45a9-4c1c-addf-8255e229008d/learning_curves.png)\n")
        
    plot_learning_curves()
    print("Evaluation complete! Results saved to results/drone/phase5_summary.md")

if __name__ == '__main__':
    run_evaluation()

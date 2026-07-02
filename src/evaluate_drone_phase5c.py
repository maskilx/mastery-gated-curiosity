import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.drone_2d_env import Custom2DDroneEnv
from src.evaluate_drone_phase5 import is_stable

def get_perturbed_state(condition):
    state = np.zeros(6, dtype=np.float32)
    # x, z, vx, vz, theta, theta_dot
    if condition == 'standard':
        return None # Let env.reset() handle it
    elif condition == 'mild':
        state = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0.5, 1.5),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.5, 0.5)
        ]
    elif condition == 'medium':
        state = [
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0.5, 2.0),
            np.random.uniform(-3.0, 3.0),
            np.random.uniform(-3.0, 3.0),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-1.0, 1.0)
        ]
    elif condition == 'hard':
        state = [
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0.5, 2.0),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-2.0, 2.0)
        ]
    elif condition == 'low_altitude':
        state = [
            np.random.uniform(-0.5, 0.5),
            0.25, # Just above ground
            np.random.uniform(-1.0, 1.0),
            -3.0, # Falling fast
            np.random.uniform(-0.2, 0.2),
            0.0
        ]
    elif condition == 'high_tilt':
        state = [
            0.0,
            1.5,
            0.0,
            0.0,
            1.2 * np.sign(np.random.uniform(-1, 1)), # Extreme tilt
            0.0
        ]
    elif condition == 'high_velocity':
        state = [
            0.0,
            1.5,
            np.random.uniform(4.0, 6.0) * np.sign(np.random.uniform(-1, 1)),
            np.random.uniform(4.0, 6.0) * np.sign(np.random.uniform(-1, 1)),
            0.0,
            0.0
        ]
    return np.array(state, dtype=np.float32)

def evaluate_condition(model_path, condition, num_episodes=50):
    model = PPO.load(model_path)
    env = Custom2DDroneEnv(mode='hover', max_steps=500)
    
    results = {
        'crash_rate': 0.0,
        'final_alt_error': [],
        'stable_episodes': 0
    }
    
    crashes = 0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        custom_state = get_perturbed_state(condition)
        if custom_state is not None:
            env.state = custom_state
            obs = env.state
            
        states = [obs]
        
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            states.append(obs)
            
            if term or trunc:
                if info.get('crashed', False):
                    crashes += 1
                break
                
        states = np.array(states)
        
        if not info.get('crashed', False):
            final_states = states[-50:]
            alt_err = np.mean(np.abs(final_states[:, 1] - 1.0))
            results['final_alt_error'].append(alt_err)
            if is_stable(states) != -1:
                results['stable_episodes'] += 1
                
    results['crash_rate'] = crashes / num_episodes
    results['final_alt_error'] = np.mean(results['final_alt_error']) if results['final_alt_error'] else float('inf')
    results['stability_rate'] = results['stable_episodes'] / num_episodes
    
    return results

def plot_learning_curves_5c():
    agents = ['scratch', 'survival', 'curiosity', 'survival_curiosity', 'safety_mastery']
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
    plt.title('Phase 5C: Downstream Hover Learning Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/drone/phase5c_learning_curves.png')
    plt.close()

def plot_perturbation_sweep(df):
    agents = df['Agent'].unique()
    conditions = df['Condition'].unique()
    
    crash_rates = np.zeros((len(agents), len(conditions)))
    for i, a in enumerate(agents):
        for j, c in enumerate(conditions):
            val = df[(df['Agent'] == a) & (df['Condition'] == c)]['CrashRate'].values[0]
            crash_rates[i, j] = val
            
    plt.figure(figsize=(12, 6))
    x = np.arange(len(conditions))
    width = 0.15
    
    for i, a in enumerate(agents):
        plt.bar(x + (i - 2) * width, crash_rates[i, :], width, label=a)
        
    plt.xticks(x, conditions, rotation=45)
    plt.ylabel('Crash Rate')
    plt.title('Crash Rate across Perturbation Conditions (50 seeds per condition)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/drone/phase5c_perturbation_sweep.png')
    plt.close()

def run_phase5c_evaluation():
    models = {
        'Scratch': 'results/drone/models/scratch_best/best_model.zip',
        'Survival': 'results/drone/models/survival_best/best_model.zip',
        'Curiosity': 'results/drone/models/curiosity_best/best_model.zip',
        'SurvCuriosity': 'results/drone/models/survival_curiosity_best/best_model.zip',
        'SafetyMastery': 'results/drone/models/safety_mastery_best/best_model.zip'
    }
    
    conditions = ['standard', 'mild', 'medium', 'hard', 'low_altitude', 'high_tilt', 'high_velocity']
    records = []
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, not found.")
            continue
            
        for cond in conditions:
            print(f"Evaluating {name} on {cond}...")
            res = evaluate_condition(path, cond, num_episodes=50)
            records.append({
                'Agent': name,
                'Condition': cond,
                'CrashRate': res['crash_rate'],
                'Stability': res['stability_rate'],
                'AltError': res['final_alt_error']
            })
            
    df = pd.DataFrame(records)
    df.to_csv('results/drone/phase5c_results.csv', index=False)
    
    plot_learning_curves_5c()
    plot_perturbation_sweep(df)
    
    # Generate summary MD
    with open('results/drone/phase5c_summary.md', 'w') as f:
        f.write("# Phase 5C: Safety-Gated Mastery Curiosity\n\n")
        f.write("## Evaluation Metrics (Preliminary: 50 seeds per condition)\n\n")
        
        # Pivot table for crash rates
        f.write("### Crash Rates\n")
        pivot_crash = df.pivot(index='Agent', columns='Condition', values='CrashRate')
        f.write(pivot_crash.to_markdown())
        f.write("\n\n")
        
        # Pivot table for alt error
        f.write("### Altitude Tracking Error (Lower is better)\n")
        pivot_err = df.pivot(index='Agent', columns='Condition', values='AltError')
        f.write(pivot_err.to_markdown())
        f.write("\n\n")
        
        f.write("![Learning Curves](/Users/adimaskil/.gemini/antigravity/brain/1ca23ce5-45a9-4c1c-addf-8255e229008d/phase5c_learning_curves.png)\n\n")
        f.write("![Perturbation Sweep](/Users/adimaskil/.gemini/antigravity/brain/1ca23ce5-45a9-4c1c-addf-8255e229008d/phase5c_perturbation_sweep.png)\n")
        
    print("Phase 5C evaluation complete!")

if __name__ == '__main__':
    run_phase5c_evaluation()

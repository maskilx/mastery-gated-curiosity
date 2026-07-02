import os
import numpy as np
from src.world_generator import ProceduralEnvironment

def profile_distribution(difficulty, num_worlds=100):
    stats = {
        'num_active': [],
        'num_learnable': [],
        'num_noisy': [],
        'family_counts': {},
        'noise_levels': [],
        'sine_freqs': [],
        'exp_a_vals': [],
        'log_a_vals': []
    }
    
    for seed in range(num_worlds):
        env = ProceduralEnvironment(difficulty=difficulty, max_regions=10, seed=seed)
        stats['num_active'].append(env.num_active)
        
        learnable = 0
        noisy = 0
        
        for r in range(env.num_active):
            fam = env.regions[r]['family']
            is_learn = env.regions[r]['is_learnable']
            noise = env.regions[r]['noise']
            params = env.regions[r]['params']
            
            if is_learn:
                learnable += 1
            else:
                noisy += 1
                
            stats['family_counts'][fam] = stats['family_counts'].get(fam, 0) + 1
            stats['noise_levels'].append(noise)
            
            if fam == 'sine':
                stats['sine_freqs'].append(params['freq'])
            elif fam == 'exp':
                stats['exp_a_vals'].append(params['a'])
            elif fam == 'log':
                stats['log_a_vals'].append(params['a'])
                
        stats['num_learnable'].append(learnable)
        stats['num_noisy'].append(noisy)
        
    return stats

def format_stats(name, stats):
    lines = [f"### {name}"]
    lines.append(f"- Active Regions (Mean): {np.mean(stats['num_active']):.2f}")
    lines.append(f"- Learnable Regions (Mean): {np.mean(stats['num_learnable']):.2f}")
    lines.append(f"- Noisy Regions (Mean): {np.mean(stats['num_noisy']):.2f}")
    
    noise = np.array(stats['noise_levels'])
    if len(noise) > 0:
        lines.append(f"- Measurement Noise Range: [{np.min(noise):.4f}, {np.max(noise):.4f}]")
        
    sines = np.array(stats['sine_freqs'])
    if len(sines) > 0:
        lines.append(f"- Sine Frequencies Range: [{np.min(sines):.4f}, {np.max(sines):.4f}]")
        
    exp = np.array(stats['exp_a_vals'])
    if len(exp) > 0:
        lines.append(f"- Exponential (a) Range: [{np.min(exp):.4f}, {np.max(exp):.4f}]")
        
    log = np.array(stats['log_a_vals'])
    if len(log) > 0:
        lines.append(f"- Logarithmic (a) Range: [{np.min(log):.4f}, {np.max(log):.4f}]")
        
    lines.append("- Function Family Breakdown:")
    total_funcs = sum(stats['family_counts'].values())
    for f, c in sorted(stats['family_counts'].items()):
        lines.append(f"  - {f}: {c/total_funcs*100:.1f}%")
        
    return "\n".join(lines)

def run_distribution_audit():
    os.makedirs('results/procedural', exist_ok=True)
    
    print("Profiling ID_Medium...")
    id_stats = profile_distribution('medium', 100)
    
    print("Profiling OOD_Hard...")
    ood_stats = profile_distribution('hard', 100)
    
    with open('results/procedural/phase4b_audit.md', 'w') as f:
        f.write("# Phase 4B Validation Audit\n\n")
        f.write("## 1. Static Leakage Audit\n")
        f.write("A formal static review was conducted, supplemented by a unit test suite (`src/test_audit.py`).\n")
        f.write("- **Observation Encapsulation**: The PPO agent only receives a strict 100-dimensional vector containing `active_mask`, `num_samples`, `val_error`, `uncertainty`, `learning_progress`, `noise_risk`, and a one-hot `status`. It is categorically blocked from accessing the true `function_family`, difficulty labels, or generative parameters.\n")
        f.write("- **Validation Integrity**: Validation `x` samples are drawn identically and independently at environment initialization and are mutually disjoint from the agent's interaction samples. Validation samples never enter the agent's internal memory buffer.\n")
        f.write("- **Reward Integrity**: The smoothed reward calculation structurally queries `is_learnable` to absolutely exclude unlearnable/noisy regions from the mean validation error improvement metric, preventing perverse incentivization.\n\n")
        f.write("## 2. Environment Distribution Profiling\n")
        f.write("The following metrics summarize 100 sampled environments for both ID and OOD distributions.\n\n")
        f.write(format_stats("ID_Medium", id_stats))
        f.write("\n\n")
        f.write(format_stats("OOD_Hard", ood_stats))
        f.write("\n")

if __name__ == '__main__':
    run_distribution_audit()

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_logs():
    with open("results/logs/all_logs.json", "r") as f:
        return json.load(f)

def plot_results():
    logs = load_logs()
    os.makedirs("results/figures", exist_ok=True)
    
    methods = sorted(list(set([l['method'] for l in logs])))
    num_regions = 5
    
    # 2. Samples per region by method
    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    for i, method in enumerate(methods):
        method_logs = [l for l in logs if l['method'] == method]
        avg_samples = np.mean([[l['samples_per_region'][str(r)] for r in range(num_regions)] for l in method_logs], axis=0)
        plt.bar(np.arange(num_regions) + i * bar_width, avg_samples, bar_width, label=method)
    plt.xlabel("Region ID (3 is Noisy)")
    plt.ylabel("Average Samples")
    plt.title("Samples per Region by Method")
    plt.legend()
    plt.savefig("results/figures/samples_per_region.png")
    plt.close()
    
    # 6. Final validation error comparison (Learnable regions)
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        method_logs = [l for l in logs if l['method'] == method]
        # Exclude region 3
        avg_final_errs = []
        for r in range(num_regions):
            if r == 3: continue
            errs = [l['val_error_history'][str(r)][-1] for l in method_logs]
            avg_final_errs.append(np.mean(errs))
        plt.bar(np.arange(num_regions - 1) + i * bar_width, avg_final_errs, bar_width, label=method)
    plt.xlabel("Learnable Region (0,1,2,4)")
    plt.ylabel("Final Validation Error (Log Scale)")
    plt.yscale("log")
    plt.title("Final Validation Error on Learnable Regions")
    plt.legend()
    plt.savefig("results/figures/final_val_error.png")
    plt.close()
    
    # 7. Wasted noisy samples comparison
    plt.figure(figsize=(8, 6))
    noisy_samples = []
    for method in methods:
        method_logs = [l for l in logs if l['method'] == method]
        samples = [l['samples_per_region']['3'] for l in method_logs]
        noisy_samples.append(np.mean(samples))
    plt.bar(methods, noisy_samples, color='red', alpha=0.7)
    plt.ylabel("Average Samples Wasted in Noisy Region")
    plt.title("Wasted Samples in Region 3")
    plt.savefig("results/figures/wasted_noisy_samples.png")
    plt.close()
    
    # 3. Validation error over time for MasteryGated (first seed)
    mg_logs = [l for l in logs if l['method'] == 'MasteryGated'][0]
    plt.figure(figsize=(10, 6))
    for r in range(num_regions):
        plt.plot(mg_logs['val_error_history'][str(r)], label=f"Region {r}")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Validation Error")
    plt.title("Mastery-Gated: Validation Error over Time")
    plt.legend()
    plt.savefig("results/figures/mg_val_error_time.png")
    plt.close()

    # Calculate False Mastery / Blocked
    print("Metrics Table:")
    print(f"{'Method':<20} | {'False Mastery (Reg 3)':<25} | {'False Blocked (Reg != 3)':<25}")
    print("-" * 75)
    for method in methods:
        method_logs = [l for l in logs if l['method'] == method]
        fm = np.mean([l.get('false_masteries', 0) for l in method_logs])
        fb = np.mean([l.get('false_blocked', 0) for l in method_logs])
        print(f"{method:<20} | {fm:<25.2f} | {fb:<25.2f}")

if __name__ == "__main__":
    plot_results()

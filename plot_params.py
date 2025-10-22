import json
import matplotlib.pyplot as plt
import numpy as np
import os

with open('runs.json', 'r') as f:
    runs = json.load(f)

print(f"Loaded {len(runs)} runs from runs.json")

os.makedirs('plots', exist_ok=True)

hyperparams = ['hidden_width', 'num_hidden_layers', 'activation_fn', 
               'learning_rate', 'dropout_rate', 'batch_size', 'weight_decay']

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, param in enumerate(hyperparams):
    ax = axes[idx]
    
    unique_values = sorted(list(set(run[param] for run in runs)))
    
    data_by_value = {val: [] for val in unique_values}
    
    for run in runs:
        param_value = run[param]
        rmse = run.get('test_rmse', run.get('test_rmse_normalized', 0))
        data_by_value[param_value].append(rmse)
    
    data = [data_by_value[val] for val in unique_values]
    labels = [str(val) for val in unique_values]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('red')
        mean.set_markeredgecolor('red')
        mean.set_markersize(6)
    
    for i, val in enumerate(unique_values):
        y = data_by_value[val]
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.4, s=30, color='darkblue')
    
    ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel('Test RMSE ($)', fontsize=10)
    ax.set_title(f'Impact of {param.replace("_", " ").title()}', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    if param in ['learning_rate', 'weight_decay']:
        ax.tick_params(axis='x', rotation=45)

axes[-2].remove()
axes[-1].remove()

plt.tight_layout()
plt.savefig('plots/hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: plots/hyperparameter_analysis.png")

for param in hyperparams:
    plt.figure(figsize=(10, 6))
    
    unique_values = sorted(list(set(run[param] for run in runs)))
    
    data_by_value = {val: [] for val in unique_values}
    
    for run in runs:
        param_value = run[param]
        rmse = run.get('test_rmse', run.get('test_rmse_normalized', 0))
        data_by_value[param_value].append(rmse)
    
    data = [data_by_value[val] for val in unique_values]
    labels = [str(val) for val in unique_values]
    
    medians = [np.median(d) for d in data]
    means = [np.mean(d) for d in data]
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(unique_values)))
    sorted_indices = np.argsort(medians)
    
    for i, patch in enumerate(bp['boxes']):
        color_idx = np.where(sorted_indices == i)[0][0]
        patch.set_facecolor(colors[color_idx])
        patch.set_alpha(0.7)
    
    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('red')
        mean.set_markeredgecolor('darkred')
        mean.set_markersize(8)
    
    for i, val in enumerate(unique_values):
        y = data_by_value[val]
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.5, s=40, color='navy', edgecolors='white', linewidths=0.5)
    
    for i, (mean_val, median_val) in enumerate(zip(means, medians)):
        plt.text(i+1, plt.ylim()[1] * 0.95, f'μ=${mean_val:,.0f}\nM=${median_val:,.0f}', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    plt.xlabel(param.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    plt.ylabel('Test RMSE ($)', fontsize=12)
    plt.title(f'Impact of {param.replace("_", " ").title()} on Model Performance', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    if param in ['learning_rate', 'weight_decay', 'dropout_rate']:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'plots/hyperparam_{param}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: plots/hyperparam_{param}.png")

print("\n" + "="*80)
print("HYPERPARAMETER IMPACT SUMMARY")
print("="*80)

for param in hyperparams:
    print(f"\n{param.replace('_', ' ').title()}:")
    print("-" * 60)
    
    unique_values = sorted(list(set(run[param] for run in runs)))
    data_by_value = {val: [] for val in unique_values}
    
    for run in runs:
        param_value = run[param]
        rmse = run.get('test_rmse', run.get('test_rmse_normalized', 0))
        data_by_value[param_value].append(rmse)
    
    stats = []
    for val in unique_values:
        rmse_values = data_by_value[val]
        stats.append({
            'value': val,
            'count': len(rmse_values),
            'mean': np.mean(rmse_values),
            'median': np.median(rmse_values),
            'std': np.std(rmse_values),
            'min': np.min(rmse_values),
            'max': np.max(rmse_values)
        })
    
    stats.sort(key=lambda x: x['median'])
    
    print(f"{'Value':<15} {'Count':>5} {'Mean RMSE':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    for s in stats:
        print(f"{str(s['value']):<15} {s['count']:>5} ${s['mean']:>11,.0f} ${s['median']:>11,.0f} ${s['std']:>11,.0f} ${s['min']:>11,.0f} ${s['max']:>11,.0f}")

print("\n" + "="*80)
print("All plots saved to 'plots/' directory")
print("="*80)

import json
import matplotlib.pyplot as plt
import numpy as np
import os

with open('runs.json', 'r') as f:
    runs = json.load(f)

print(f"Loaded {len(runs)} runs from runs.json")

os.makedirs('plots', exist_ok=True)

all_losses = []
for run in runs:
    all_losses.extend(run['epoch_losses'])

min_loss = min(all_losses)
max_loss = max(all_losses)
y_margin = (max_loss - min_loss) * 0.1
y_min = min_loss - y_margin
y_max = max_loss + y_margin

print(f"Loss range: {min_loss:,.2f} to {max_loss:,.2f}")
print(f"Y-axis limits: {y_min:,.2f} to {y_max:,.2f}")
print(f"\nGenerating plots...")

for i, run in enumerate(runs):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(run['epoch_losses']) + 1)
    plt.plot(epochs, run['epoch_losses'], 'b-', linewidth=2, marker='o', markersize=4)
    
    plt.ylim(y_min, y_max)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    title = f"Run {i+1}: "
    title += f"Width={run['hidden_width']}, Layers={run['num_hidden_layers']}, "
    title += f"Act={run['activation_fn']}\n"
    title += f"LR={run['learning_rate']}, Dropout={run['dropout_rate']}, "
    title += f"Batch={run['batch_size']}, WD={run['weight_decay']}\n"
    title += f"Test RMSE: ${run['test_rmse']:,.2f}"
    plt.title(title, fontsize=10)
    
    filename = f"plots/run_{i+1:03d}_rmse_{run['test_rmse']:.0f}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"  Saved: {filename}")

print(f"\nDone! Generated {len(runs)} plots in 'plots/' directory")

plt.figure(figsize=(14, 8))
for i, run in enumerate(runs):
    epochs = range(1, len(run['epoch_losses']) + 1)
    plt.plot(epochs, run['epoch_losses'], alpha=0.5, linewidth=1, label=f"Run {i+1} (RMSE: ${run['test_rmse']:,.0f})")

plt.ylim(y_min, y_max)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('All Runs Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('plots/all_runs_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: plots/all_runs_comparison.png")

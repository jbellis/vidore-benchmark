import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('ocr.json', 'r') as file:
    data = json.load(file)

# Prepare data for plotting
datasets = list(data.keys())
methods = ['llamaparse', 'unstructured', 'flash']
metrics = ['bge_m3', 'bm25']

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
fig.suptitle('OCR Comparison: bge_m3 vs bm25', fontsize=16)

# Set width of bars
bar_width = 0.25

# Set positions of the bars on X axis
r1 = np.arange(len(datasets))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Colors for each method
colors = ['#8884d8', '#82ca9d', '#ffc658']

# Create bars for each metric
for i, metric in enumerate(metrics):
    ax = ax1 if i == 0 else ax2
    
    for j, method in enumerate(methods):
        values = [data[dataset].get(f'{method}_{metric}', 0) for dataset in datasets]
        ax.bar(eval(f'r{j+1}'), values, width=bar_width, color=colors[j], label=f'{method.capitalize()} ({metric})')

    # Add labels and title
    ax.set_xlabel('Datasets')
    ax.set_ylabel('NDCG@5 Recall')
    ax.set_title(f'{metric.upper()} Comparison')
    ax.set_xticks([r + bar_width for r in range(len(datasets))])
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('ocr.png', dpi=300, bbox_inches='tight')

print("The plot has been saved as 'ocr.png'.")

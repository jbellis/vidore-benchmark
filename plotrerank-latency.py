import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['rrf', 'jina', 'cohere', 'voyage', 'voyage-lite', 'bge_3090']
times = {
    'rrf': 21,
    'jina': 230,
    'cohere': 90,
    'voyage': 311,
    'voyage-lite': 240,
    'bge_3090': 115,
}

# Calculate speeds
startup = 20
count = 500
speeds = {model: (times[model] - startup)/count for model in models}

# Colors
colors = {
    'rrf': '#ff7f0e',      # orange
    'jina': '#9467bd',     # purple
    'cohere': '#1f77b4',   # blue
    'voyage': '#8c564b',   # brown
    'voyage-lite': '#e377c2',  # pink
    'bge_3090': '#2ca02c',     # green
}

# Create the plot
plt.figure(figsize=(10, 6))

# Create bars using the specified order
y_pos = np.arange(len(models))
speeds_ordered = [speeds[model] for model in models]
colors_ordered = [colors[model] for model in models]

bars = plt.bar(y_pos, speeds_ordered, color=colors_ordered)

# Customize the plot
plt.xlabel('Models')
plt.ylabel('Latency (seconds per request)')
plt.title('Reranking Speed')
plt.xticks(y_pos, models, rotation=45)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2g}s',
             ha='center', va='bottom')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('rerank-latency.png', dpi=300, bbox_inches='tight')

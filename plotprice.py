import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['rrf', 'jina', 'cohere', 'voyage', 'voyage-lite']

# Usage data
searches = 2971
tokens = 71801441

# Calculate prices
prices = {
    'rrf': 0,  # Free
    'jina': (tokens / 1_000_000) * 0.020,  # $0.020 per 1M tokens
    'cohere': searches * (2 / 1000),  # $2 per 1k searches
    'voyage': (tokens / 1_000_000) * 0.05,  # $0.05 per 1M tokens
    'voyage-lite': (tokens / 1_000_000) * 0.02,  # $0.02 per 1M tokens
}

# Colors (same as previous)
colors = {
    'rrf': '#ff7f0e',      # orange
    'jina': '#9467bd',     # purple
    'cohere': '#1f77b4',   # blue
    'voyage': '#8c564b',   # brown
    'voyage-lite': '#e377c2'  # pink
}

# Create the plot
plt.figure(figsize=(10, 6))

# Create bars using the specified order
y_pos = np.arange(len(models))
prices_ordered = [prices[model] for model in models]
colors_ordered = [colors[model] for model in models]

bars = plt.bar(y_pos, prices_ordered, color=colors_ordered)

# Customize the plot
plt.xlabel('Models')
plt.ylabel('Cost (USD)')
plt.title('Reranking Cost Comparison')
plt.xticks(y_pos, models, rotation=45)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.2f}',
             ha='center', va='bottom')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print actual values for verification
print("\nDetailed costs:")
for model in models:
    print(f"{model}: ${prices[model]:.2f}")

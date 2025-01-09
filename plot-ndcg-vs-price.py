import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    'voyage_3_large', 'openai_v3_large', 'nvidia_llama_v1*', 'voyage_3_lite',
    'stella_1_5b*', 'stella*', 'openai_v3_small', 'cohere_v3', 'jina_v3', 'gemini_004',
    'modernbert_embed*'
]

ndcg = [0.990, 0.874, 0.875, 0.858, 0.844, 0.839, 0.821, 0.767, 0.694, 0.804, 0.750]
price = [0.18, 0.13, 0.035, 0.02, 0.054, 0.015, 0.01, 0.1, 0.02, 0, 0.0052]

# Create figure and axis
plt.figure(figsize=(12, 8))

# Create scatter plot
plt.scatter(price, ndcg, alpha=0.6)

# Add labels for each point
for i, model in enumerate(models):
    plt.annotate(model, (price[i], ndcg[i]), 
                xytext=(5, 5), textcoords='offset points')

# Customize plot
plt.xlabel('Price per Million Tokens ($)', fontsize=12)
plt.ylabel('Normalized Average NDCG@5', fontsize=12)
plt.title('Model Performance vs Cost', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend with asterisk explanation
plt.figtext(0.99, 0.01, '* = estimated price', 
            ha='right', va='bottom', fontsize=10, style='italic')
plt.tight_layout()
filename = 'ndcg-vs-price.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {filename}")
plt.show()

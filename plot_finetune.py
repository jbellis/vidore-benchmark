import matplotlib.pyplot as plt

# Define model groups and their colors
MODEL_GROUPS = {
    'GTE Models': ['gte-large', 'gte-large-finetuned'],
    'OpenAI Models': ['openai-v3-small', 'openai-v3-large'],
    'Stella-400M Models': ['stella-400M', 'stella-400M-finetuned'],
    'Stella-1.5B Models': ['stella-1.5B', 'stella-1.5B-finetuned']
}

# Color palette for groups - using similar colors within groups
GROUP_COLORS = {
    'GTE Models': ['#7cc7ff', '#1f77b4'],           # Blues
    'OpenAI Models': ['#98df8a', '#2ca02c'],        # Greens
    'Stella-400M Models': ['#ffbb78', '#ff7f0e'],   # Oranges
    'Stella-1.5B Models': ['#ff9896', '#d62728']    # Reds
}

# Hardcoded data
DATA = {
    'arxivqa': {
        'gte-large': 0.49,
        'stella-400M': 0.50,
        'stella-1.5B': 0.54,
        'openai-v3-small': 0.59,
        'gte-large-finetuned': 0.61,
        'openai-v3-large': 0.62,
        'stella-400M-finetuned': 0.69,
        'stella-1.5B-finetuned': 0.70
    },
    'infovqa': {
        'stella-400M': 0.82,
        'stella-400M-finetuned': 0.85,
        'stella-1.5B': 0.81,
        'stella-1.5B-finetuned': 0.86,
        'openai-v3-small': 0.77,
        'openai-v3-large': 0.80
    }
}

def plot_dataset(dataset_name):
    x = range(1)  # Only one dataset per plot
    group_width = 0.15  # Width for each group
    group_spacing = 0.1  # Additional space between groups
    bar_width = group_width / 2  # Width for each bar within group
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for group_idx, (group_name, models) in enumerate(MODEL_GROUPS.items()):
        group_x = [xi + group_idx * (group_width + group_spacing) for xi in x]
        
        for model_idx, model in enumerate(models):
            if model in DATA[dataset_name]:  # Only plot if model exists for this dataset
                value = DATA[dataset_name][model]
                bar_x = [x + model_idx * bar_width for x in group_x]
                
                bars = ax.bar(bar_x, [value], bar_width,
                             label=model,
                             color=GROUP_COLORS[group_name][model_idx])
                
                # Add text labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:  # Only label bars that have values
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('NDCG@5')
    ax.set_title(f'Model Performance Comparison - {dataset_name}')
    ax.set_xticks([])  # Remove x-axis ticks since we only have one dataset
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'finetune-{dataset_name}.png', bbox_inches='tight')
    plt.close()
    print(f"Graph saved as finetune-{dataset_name}.png")

def main():
    for dataset in DATA.keys():
        plot_dataset(dataset)

if __name__ == "__main__":
    main()

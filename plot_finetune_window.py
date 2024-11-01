import matplotlib.pyplot as plt

# Map models to their colors
MODEL_COLORS = {
    'stella-400M': '#ffbb78',            # Original stella-400M color
    'openai-v3-large': '#2ca02c',        # Green from original
    'stella-400M-finetuned-256': '#ffa154', # Interpolated color for 256
    'stella-400M-finetuned-1024': '#ff7f0e' # Original finetuned color
}

# Data
DATA = {
    'arxivqa': {
        'stella-400M': 0.50,
        'openai-v3-large': 0.62,
        'stella-400M-finetuned-256': 0.66,
        'stella-400M-finetuned-1024': 0.71
    }
}

def plot_dataset(dataset_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort models by their values
    data = DATA[dataset_name]
    sorted_models = sorted(data.items(), key=lambda x: x[1])
    
    # Plot bars with equal spacing
    bar_width = 0.4  # Make bars wider
    for idx, (model, value) in enumerate(sorted_models):
        bars = ax.bar([idx], [value], bar_width,
                     label=model,
                     color=MODEL_COLORS[model])
        
        # Add text labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label bars that have values
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('NDCG@5')
    ax.set_title('Projecting to lower dimensions, arxivqa')
    
    # Set x-ticks and rotated labels below bars
    ax.set_xticks(range(len(sorted_models)))
    ax.set_xticklabels([model for model, _ in sorted_models], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'finetune-window-{dataset_name}.png', bbox_inches='tight')
    plt.close()
    print(f"Graph saved as finetune-window-{dataset_name}.png")

def main():
    for dataset in DATA.keys():
        plot_dataset(dataset)

if __name__ == "__main__":
    main()

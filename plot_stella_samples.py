import matplotlib.pyplot as plt

# Data points for stella-400M fine-tuning on arxivqa
DATA = {
    'base': 0.50,
    200: 0.62,
    400: 0.63,
    800: 0.65,
    1600: 0.67,
    3200: 0.68,
    6400: 0.71
}

def plot_samples():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use same orange colors as stella-400M from other script
    colors = ['#ffbb78'] + ['#ff7f0e'] * (len(DATA) - 1)  # Base model color + finetuned color
    bars = ax.bar(range(len(DATA)), list(DATA.values()), color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom')
    
    # Customize x-axis
    ax.set_xticks(range(len(DATA)))
    ax.set_xticklabels(DATA.keys(), rotation=45)
    
    # Labels and title
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('NDCG@5')
    ax.set_title('Stella-400M Fine-tuning Performance vs Training Samples')
    
    plt.tight_layout()
    plt.savefig('stella-samples.png')
    plt.close()
    print("Graph saved as stella-samples.png")

def main():
    plot_samples()

if __name__ == "__main__":
    main()

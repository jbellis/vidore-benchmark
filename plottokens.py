import os
import json
import matplotlib.pyplot as plt

# Define a color for openai_v3_small
COLOR_OPENAI = '#65ff3d'  # light green

# Token counts for each dataset
TOKEN_COUNTS = {
    'infovqa_test_subsampled': 657.16,
    'docvqa_test_subsampled': 375.00,
    'tatdqa_test': 783.72,
    'shiftproject_test': 1150.22,
    'tabfquad_test_subsampled': 395.90,
    'arxivqa_test_subsampled': 138.18
}

def extract_dataset(filename):
    parts = filename.split('_')
    return '_'.join(parts[1:-2])  # Assuming the format is 'vidore_dataset_openai_v3_small.pth'

def read_ndcg_value(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]['ndcg_at_5']

def main():
    output_dir = 'outputs'
    data = {}

    for filename in os.listdir(output_dir):
        if filename.startswith('vidore_') and filename.endswith('openai_v3_small.pth'):
            dataset = extract_dataset(filename)
            file_path = os.path.join(output_dir, filename)
            ndcg_value = read_ndcg_value(file_path)
            data[dataset] = ndcg_value

    # Prepare data for plotting
    datasets = list(data.keys())
    ndcg_values = [data[dataset] for dataset in datasets]
    token_counts = [TOKEN_COUNTS[dataset] for dataset in datasets]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot NDCG values
    x = range(len(datasets))
    bars = ax1.bar(x, ndcg_values, color=COLOR_OPENAI, alpha=0.7)
    ax1.set_ylabel('NDCG@5', color=COLOR_OPENAI)
    ax1.tick_params(axis='y', labelcolor=COLOR_OPENAI)

    # Add text labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=8, color=COLOR_OPENAI)

    # Create a second y-axis for token counts
    ax2 = ax1.twinx()
    ax2.plot(x, token_counts, color='red', marker='o', linestyle='-', linewidth=2, markersize=8)
    ax2.set_ylabel('Average Token Count', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add token count labels
    for i, count in enumerate(token_counts):
        ax2.text(i, count, f'{count:.2f}', ha='center', va='bottom', fontsize=8, color='red')

    plt.title('NDCG@5 and Average Token Count by Dataset (openai_v3_small)')
    plt.xticks(x, datasets, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('ndcg_token_comparison.png')
    plt.show()
    print("Graph saved as ndcg_token_comparison.png and displayed")

if __name__ == "__main__":
    main()

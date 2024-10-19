import os
import json
import matplotlib.pyplot as plt

# Define a color palette
COLOR_PALETTE = {
    'cohere': '#1f77b4',  # blue
    'rrf': '#ff7f0e',  # orange
}

def extract_dataset_and_rerank_type(filename):
    # 1. Split off cohere / rrf as rerank type
    if filename.endswith('_cohere.pth'):
        rerank_type = 'cohere'
        parts = filename[:-11].split('_')  # Remove '_cohere.pth'
    elif filename.endswith('_rrf.pth'):
        rerank_type = 'rrf'
        parts = filename[:-8].split('_')  # Remove '_rrf.pth'
    else:
        return None, None

    print(f"Extracted rerank type: {rerank_type} from filename {filename}")
    # 2. Check if 'best' is in filename
    if 'best' not in parts:
        print(f"Warning: 'best' not found in filename {filename}. Skipping.")
        return None, None

    # 3. Split off 'best'
    best_index = parts.index('best')
    parts_before_best = parts[:best_index]

    # 4. Split off the last fragment after 'best'
    parts_before_best = parts_before_best[:-1]

    # 5. Split off 'vidore' from the front
    if parts_before_best[0] == 'vidore':
        parts_before_best = parts_before_best[1:]

    # 6. The rest is the dataset name
    dataset = '_'.join(parts_before_best)

    return dataset, rerank_type

def read_ndcg_value(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]['ndcg_at_5']

def main():
    output_dir = 'outputs'
    rerank_types = ['cohere', 'rrf']
    data = {}

    for filename in os.listdir(output_dir):
        if filename.startswith('vidore_') and filename.endswith('.pth'):
            dataset, rerank_type = extract_dataset_and_rerank_type(filename)
            if dataset and rerank_type:
                file_path = os.path.join(output_dir, filename)
                ndcg_value = read_ndcg_value(file_path)

                if dataset not in data:
                    data[dataset] = {}
                data[dataset][rerank_type] = ndcg_value

    # Prepare data for plotting
    datasets = list(data.keys())
    x = range(len(datasets))
    width = 0.35  # Width of each bar

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, rerank_type in enumerate(rerank_types):
        values = [data[dataset].get(rerank_type, 0) for dataset in datasets]
        bars = ax.bar([xi + i * width for xi in x], values, width, label=rerank_type, color=COLOR_PALETTE[rerank_type])

        # Add text labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('NDCG@5')
    ax.set_title('NDCG@5 by Dataset and Rerank Type')
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('rerank_comparison.png')
    plt.show()
    print("Graph saved as rerank_comparison.png and displayed")

if __name__ == "__main__":
    main()

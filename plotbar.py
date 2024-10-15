import os
import json
import glob
import matplotlib.pyplot as plt

# Define a color palette
COLOR_PALETTE = {
    'stella': '#1f77b4',  # blue
    'gemini_004': '#ff7f0e',  # orange
    'openai_v3_small': '#65ff3d',  # light green (50% more saturated)
    'openai_v3_large': '#2ca02c',  # green
    'colbert_fast': '#ff4d4a',  # light red (50% more saturated)
    'colbert_best': '#d62728',  # red
}

def extract_dataset_and_model(filename):
    parts = filename.split('_')
    models = {
        'stella': 'stella',
        'gemini_004': 'gemini_004',
        'openai_v3_small': 'openai_v3_small',
        'openai_v3_large': 'openai_v3_large'
    }
    
    for model in models:
        if filename.endswith(f"{model}.pth"):
            dataset = '_'.join(parts[1:-len(model.split('_'))])
            return dataset, models[model]
    
    return None, None

def read_ndcg_value(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]['ndcg_at_5']

def find_best_colbert_live_ndcg(dataset, fast=False):
    pattern = f'outputs-colbert-live/vidore_{dataset}*.pth'
    files = glob.glob(pattern)
    best_ndcg = 0
    best_elapsed = float('inf')
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        key = list(data.keys())[0]
        ndcg = data[key]['ndcg_at_5']
        elapsed = data[key]['elapsed']
        
        if not fast:
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_elapsed = elapsed
        else:
            if ndcg > best_ndcg and elapsed <= 0.2 * best_elapsed:
                best_ndcg = ndcg
                best_elapsed = elapsed
    
    return best_ndcg

def main():
    output_dir = 'outputs'
    models = ['stella', 'gemini_004', 'openai_v3_small', 'openai_v3_large', 'colbert_fast', 'colbert_best']
    data = {}

    for filename in os.listdir(output_dir):
        if filename.startswith('vidore_') and filename.endswith('.pth'):
            dataset, model = extract_dataset_and_model(filename)
            if dataset and model:
                file_path = os.path.join(output_dir, filename)
                ndcg_value = read_ndcg_value(file_path)
                
                if dataset not in data:
                    data[dataset] = {}
                data[dataset][model] = ndcg_value
            else:
                print(f"Warning: Unable to extract dataset and model from file '{filename}'. Skipping this file.")

    # Add colbert_best and colbert_fast data
    for dataset in data.keys():
        best_colbert_ndcg = find_best_colbert_live_ndcg(dataset)
        data[dataset]['colbert_best'] = best_colbert_ndcg
        fast_colbert_ndcg = find_best_colbert_live_ndcg(dataset, fast=True)
        data[dataset]['colbert_fast'] = fast_colbert_ndcg

    # Prepare data for plotting
    datasets = list(data.keys())
    x = range(len(datasets))
    width = 0.15  # Reduced width to accommodate 5 bars

    fig, ax = plt.subplots(figsize=(24, 12))

    for i, model in enumerate(models):
        values = [data[dataset].get(model, 0) for dataset in datasets]
        bars = ax.bar([xi + i * width for xi in x], values, width, label=model, color=COLOR_PALETTE[model])
        
        # Add text labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('NDCG@5')
    ax.set_title('NDCG@5 by Dataset and Model')
    ax.set_xticks([xi + 2.5 * width for xi in x])  # Adjusted to center x-axis labels
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('ndcg_comparison.png')
    plt.show()
    print("Graph saved as ndcg_comparison.png and displayed")

if __name__ == "__main__":
    main()

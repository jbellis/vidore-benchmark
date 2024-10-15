import os
import json
import glob
import matplotlib.pyplot as plt

def extract_dataset_and_model(filename):
    parts = filename.split('_')
    models = {
        'stella': 'stella',
        'gemini_004': 'gemini_004',
        'openai_v3_small': 'openai_v3_small'
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
    models = ['stella', 'gemini_004', 'openai_v3_small', 'colbert_best', 'colbert_fast']
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
        ax.bar([xi + i * width for xi in x], values, width, label=model)

    ax.set_ylabel('NDCG@5')
    ax.set_title('NDCG@5 by Dataset and Model')
    ax.set_xticks([xi + 2 * width for xi in x])
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('ndcg_comparison.png')
    print("Graph displayed and saved as ndcg_comparison.png")

if __name__ == "__main__":
    main()

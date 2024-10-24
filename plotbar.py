import os
import json
import glob
import random
import matplotlib.pyplot as plt

# Define a color palette with slightly more saturated colors for some models
MODEL_COLORS = {
    'stella': '#a6cee3',  # light blue
    'gemini_004': '#fdbf6f',  # light orange
    'openai_v3_small': '#95d679',  # slightly more saturated light green
    'openai_v3_large': '#33a02c',  # darker green
    'bge_m3': '#b894c2',  # slightly more saturated light purple
    'bm25': '#b15928',  # brown
    'gte_large': '#f98080',  # slightly more saturated light pink
}

def extract_dataset_and_model(filename):
    parts = filename.split('_')

    for model in MODEL_COLORS.keys():
        if filename.endswith(f"{model}.pth"):
            dataset = '_'.join(parts[1:-len(model.split('_'))])
            return dataset, model
    
    return None, None

def read_ndcg_value(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]['ndcg_at_5']

def main():
    output_dir = 'outputs-dpr'
    models = MODEL_COLORS.keys()
    data = {}

    for filename in os.listdir(output_dir):
        if filename.startswith('vidore_') and 'flash' in filename and filename.endswith('.pth'):
            dataset, model = extract_dataset_and_model(filename)
            if dataset and model:
                file_path = os.path.join(output_dir, filename)
                ndcg_value = read_ndcg_value(file_path)
                
                if dataset not in data:
                    data[dataset] = {}
                data[dataset][model] = ndcg_value
            else:
                print(f"Warning: Unable to extract dataset and model from file '{filename}'. Skipping this file.")

    # Prepare data for plotting
    datasets = list(data.keys())
    x = range(len(datasets))
    
    width = 0.1125  # Adjusted width (0.15 * 0.75)

    fig, ax = plt.subplots(figsize=(24, 12))

    for i, model in enumerate(models):
        values = [data[dataset].get(model, 0) for dataset in datasets]
        
        bars = ax.bar([xi + i * width for xi in x], values, width, label=model, color=MODEL_COLORS[model])
        
        # Add text labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('NDCG@5')
    ax.set_title('NDCG@5 by Dataset and Model')
    ax.set_xticks([xi + (len(models) - 1) * width / 2 for xi in x])  # Adjusted to center x-axis labels
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('dpr_comparison.png')
    plt.show()
    print("Graph saved as dpr_comparison.png and displayed")

if __name__ == "__main__":
    main()

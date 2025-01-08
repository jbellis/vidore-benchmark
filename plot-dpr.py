import os
import json
import glob
import random
import matplotlib.pyplot as plt
import re

# Define a color palette with slightly more saturated colors for some models
MODEL_COLORS = {
    'stella': '#a6cee3',  # light blue
    'gemini_004': '#fdbf6f',  # light orange
    'openai_v3_small': '#95d679',  # slightly more saturated light green
    'openai_v3_large': '#33a02c',  # darker green
    'voyage_3_large': '#b894c2',  # slightly more saturated light purple
    'voyage_3_lite': '#cab2d6',  # lighter purple
    'modernbert_embed': '#f98080',  # slightly more saturated light pink
    'cohere_v3': '#ff7f00',  # bright orange
    'jina_v3': '#e31a1c',    # bright red
    'nvidia_llama_v1': '#6a3d9a',  # dark purple
    'stella_1_5b': '#1f78b4',  # darker blue
}
MODEL_FAMILIES = [
    ('modernbert_embed',),
    ('jina_v3',),
    ('cohere_v3',),
    ('nvidia_llama_v1',),
    ('gemini_004',),
    ('openai_v3_large', 'openai_v3_small'),
    ('voyage_3_large', 'voyage_3_lite'),
    ('stella_1_5b', 'stella'),
]

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

def process_dataset_name(dataset):
    return re.sub(r'_test.*$', '', dataset)

def main():
    output_dir = 'outputs-dpr'
    # Flatten MODEL_FAMILIES into a list of models while preserving order
    models = [model for family in MODEL_FAMILIES for model in family]
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
    x = [i * 1.5 for i in range(len(datasets))]  # Increase spacing by 10%
    
    width = 0.1125  # Adjusted width (0.15 * 0.75)

    fig, ax = plt.subplots(figsize=(24, 12))

    current_offset = 0
    for family in MODEL_FAMILIES:
        for model in family:
            values = [data[dataset].get(model, 0) for dataset in datasets]
            
            bars = ax.bar([xi + current_offset * width for xi in x], values, width, 
                         label=model, color=MODEL_COLORS[model])
            
            # Add text labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
            current_offset += 1
        # Add space between families (reduced by 60%)
        current_offset += 0.2

    ax.set_ylabel('NDCG@5')
    ax.set_title('NDCG@5 by Dataset and Model')
    # Calculate total width including spaces between families (reduced by 60%)
    total_width = sum(len(family) for family in MODEL_FAMILIES) + (len(MODEL_FAMILIES) - 1) * 0.2
    ax.set_xticks([xi + (total_width - 1) * width / 2 for xi in x])  # Center x-axis labels
    processed_datasets = [process_dataset_name(dataset) for dataset in datasets]
    ax.set_xticklabels(processed_datasets, rotation=0, ha='center')
    ax.legend()

    plt.tight_layout()
    plt.savefig('dpr_comparison.png')
    plt.show()
    print("Graph saved as dpr_comparison.png and displayed")

if __name__ == "__main__":
    main()

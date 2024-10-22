import os
import json
import matplotlib.pyplot as plt

# Define a color palette
COLOR_PALETTE = {
    'bm25': '#808080',  # medium grey
    'dpr': '#A9A9A9',   # dark grey
    'rrf': '#ff7f0e',  # orange
    'jina': '#9467bd',  # purple
    'cohere': '#1f77b4',  # blue
    'voyage': '#8c564b',  # brown
    'voyage-lite': '#e377c2',  # pink
    'bge': '#2ca02c',  # green
}

def get_embeddings_model(dataset):
    if 'infovqa' in dataset:
        return 'stella'
    elif 'docvqa' in dataset:
        return 'openai_v3_large'
    elif 'tabfquad' in dataset:
        return 'openai_v3_large'
    elif 'shift' in dataset:
        return 'bge_m3'
    elif 'tatdqa' in dataset:
        return 'gemini_004'
    elif 'arxivqa' in dataset:
        return 'openai_v3_large'
    else:
        return None

def extract_dataset_and_rerank_type(filename):
    # 1. Split off cohere / rrf / jina / voyage / voyage-lite / bge as rerank type
    if filename.endswith('_cohere.pth'):
        rerank_type = 'cohere'
        parts = filename[:-11].split('_')  # Remove '_cohere.pth'
    elif filename.endswith('_rrf.pth'):
        rerank_type = 'rrf'
        parts = filename[:-8].split('_')  # Remove '_rrf.pth'
    elif filename.endswith('_jina.pth'):
        rerank_type = 'jina'
        parts = filename[:-9].split('_')  # Remove '_jina.pth'
    elif filename.endswith('_voyage.pth'):
        rerank_type = 'voyage'
        parts = filename[:-11].split('_')  # Remove '_voyage.pth'
    elif filename.endswith('_voyage_lite.pth'):
        rerank_type = 'voyage-lite'
        parts = filename[:-15].split('_')  # Remove '_voyage_lite.pth'
    elif filename.endswith('_bge.pth'):
        rerank_type = 'bge'
        parts = filename[:-8].split('_')  # Remove '_bge.pth'
    else:
        return None, None

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

def get_bm25_filename(rrf_filename):
    return rrf_filename.replace('best_rrf', 'bm25')

def get_dpr_filename(rrf_filename, dataset):
    embeddings_model = get_embeddings_model(dataset)
    if embeddings_model:
        return rrf_filename.replace('best_rrf', embeddings_model)
    return None

def read_ndcg_value(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]['ndcg_at_5']


def main():
    output_dir = 'outputs'
    score_types = ['bm25', 'dpr', 'rrf', 'jina', 'cohere', 'voyage', 'voyage-lite', 'bge']
    data = {}

    for filename in os.listdir(output_dir):
        if filename.startswith('vidore_') and filename.endswith('.pth'):
            dataset, rerank_type = extract_dataset_and_rerank_type(filename)
            if dataset and rerank_type == 'rrf':  # We use rrf files as a base
                rrf_file_path = os.path.join(output_dir, filename)
                if not os.path.exists(rrf_file_path):
                    print(f"Warning: RRF file not found: {rrf_file_path}")
                    continue
                rrf_ndcg = read_ndcg_value(rrf_file_path)

                cohere_file_path = os.path.join(output_dir, filename.replace('_rrf.pth', '_cohere.pth'))
                jina_file_path = os.path.join(output_dir, filename.replace('_rrf.pth', '_jina.pth'))
                voyage_file_path = os.path.join(output_dir, filename.replace('_rrf.pth', '_voyage.pth'))
                voyage_lite_file_path = os.path.join(output_dir, filename.replace('_rrf.pth', '_voyage_lite.pth'))
                bge_file_path = os.path.join(output_dir, filename.replace('_rrf.pth', '_bge.pth'))
                bm25_file_path = os.path.join(output_dir, get_bm25_filename(filename))
                dpr_filename = get_dpr_filename(filename, dataset)
                dpr_file_path = os.path.join(output_dir, dpr_filename) if dpr_filename else None

                if dataset not in data:
                    data[dataset] = {}

                data[dataset]['rrf'] = rrf_ndcg

                if os.path.exists(cohere_file_path):
                    data[dataset]['cohere'] = read_ndcg_value(cohere_file_path)
                else:
                    print(f"Warning: Cohere file not found: {cohere_file_path}")
                    data[dataset]['cohere'] = 0

                if os.path.exists(jina_file_path):
                    data[dataset]['jina'] = read_ndcg_value(jina_file_path)
                else:
                    print(f"Warning: Jina file not found: {jina_file_path}")
                    data[dataset]['jina'] = 0

                if os.path.exists(voyage_file_path):
                    data[dataset]['voyage'] = read_ndcg_value(voyage_file_path)
                else:
                    print(f"Warning: Voyage file not found: {voyage_file_path}")
                    data[dataset]['voyage'] = 0

                if os.path.exists(voyage_lite_file_path):
                    data[dataset]['voyage-lite'] = read_ndcg_value(voyage_lite_file_path)
                else:
                    print(f"Warning: Voyage-lite file not found: {voyage_lite_file_path}")
                    data[dataset]['voyage-lite'] = 0

                if os.path.exists(bge_file_path):
                    data[dataset]['bge'] = read_ndcg_value(bge_file_path)
                else:
                    print(f"Warning: BGE file not found: {bge_file_path}")
                    data[dataset]['bge'] = 0

                if os.path.exists(bm25_file_path):
                    data[dataset]['bm25'] = read_ndcg_value(bm25_file_path)
                else:
                    print(f"Warning: BM25 file not found: {bm25_file_path}")
                    data[dataset]['bm25'] = 0

                if dpr_file_path and os.path.exists(dpr_file_path):
                    data[dataset]['dpr'] = read_ndcg_value(dpr_file_path)
                else:
                    if dpr_file_path:
                        print(f"Warning: DPR file not found: {dpr_file_path}")
                    else:
                        print(f"Warning: DPR filename could not be determined for dataset: {dataset}")
                    data[dataset]['dpr'] = 0

    # Prepare data for plotting
    datasets = list(data.keys())
    x = range(len(datasets))
    width = 0.11  # Width of each bar (adjusted for 7 score types)

    fig, ax = plt.subplots(figsize=(24, 10))

    for i, score_type in enumerate(score_types):
        values = [data[dataset].get(score_type, 0) for dataset in datasets]
        bars = ax.bar([xi + i * width for xi in x], values, width, label=score_type, color=COLOR_PALETTE[score_type])

        # Add text labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('NDCG@5')
    ax.set_title('NDCG@5 by Dataset')
    ax.set_xticks([xi + (len(score_types) - 1) * width / 2 for xi in x])
    ax.set_xticklabels([dataset.split('_test')[0] for dataset in datasets], ha='center')
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Graph saved as score_comparison.png and displayed")

if __name__ == "__main__":
    main()

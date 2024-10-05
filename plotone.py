import os
import json
import matplotlib.pyplot as plt
import re

def parse_filename(filename):
    parts = filename.split('_')
    candidates = int(parts[-1].split('.')[0])
    ann = float(parts[-2])
    querypool = float(parts[-3])
    dataset_name = '_'.join(parts[:-3])
    return dataset_name, querypool, ann, candidates

def read_data(directory):
    data_points = []
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            dataset_name, querypool, ann, candidates = parse_filename(filename)
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                ndcg_at_5 = data[next(iter(data))]['ndcg_at_5']
                elapsed = data[next(iter(data))]['elapsed']
                qps = 500 / elapsed
                data_points.append((ndcg_at_5, qps, querypool, ann, candidates, dataset_name))
    return data_points

def plot_data(data_points):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ndcg_values, qps_values = zip(*[(d[0], d[1]) for d in data_points])
    
    scatter = ax.scatter(ndcg_values, qps_values)
    
    ax.set_xlabel('NDCG@5')
    ax.set_ylabel('QPS')
    ax.set_title('NDCG@5 vs QPS')
    
    for i, (ndcg, qps, querypool, ann, candidates, dataset_name) in enumerate(data_points):
        label = f"{querypool},{ann},{candidates}={ndcg:.3f}"
        ax.annotate(label, (ndcg, qps), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    directory = 'outputs'
    data_points = read_data(directory)
    fig = plot_data(data_points)
    
    # Display the plot in a window
    plt.show()
    
    # Ask user if they want to save the plot
    save_plot = input("Do you want to save the plot? (y/n): ").lower().strip()
    if save_plot == 'y':
        filename = input("Enter filename (default: ndcg_vs_qps_plot.png): ").strip()
        if not filename:
            filename = 'ndcg_vs_qps_plot.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")

if __name__ == "__main__":
    main()

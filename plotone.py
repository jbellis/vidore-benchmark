import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from adjustText import adjust_text
import argparse

def parse_filename(filename):
    parts = filename.split('_')
    candidates = int(parts[-1].split('.')[0])
    ann = float(parts[-2])
    querypool = float(parts[-3])
    dataset_name = '_'.join(parts[:-3])
    return dataset_name, querypool, ann, candidates

def read_data(directory, dataset_filter=None):
    data_points = []
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            dataset_name, querypool, ann, candidates = parse_filename(filename)
            if dataset_filter and dataset_filter not in dataset_name:
                continue
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                ndcg_at_5 = data[next(iter(data))]['ndcg_at_5']
                elapsed = data[next(iter(data))]['elapsed']
                qps = 500 / elapsed
                data_points.append((ndcg_at_5, qps, querypool, ann, candidates, dataset_name))
    return data_points

def is_pareto_efficient(points):
    is_efficient = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] > point, axis=1)
            is_efficient[i] = True
    return is_efficient

def plot_data(data_points, plot_all=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract dataset name from the first data point
    dataset_name = data_points[0][5]
    
    # Group data points by querypool
    grouped_data = defaultdict(list)
    for point in data_points:
        grouped_data[point[2]].append(point)
    
    # Generate a color map
    color_map = plt.cm.viridis
    colors = [color_map(i) for i in np.linspace(0, 1, len(grouped_data))]
    
    texts = []
    for (querypool, points), color in zip(grouped_data.items(), colors):
        ndcg_values, qps_values = zip(*[(p[0], p[1]) for p in points])
        
        # Find Pareto optimal points
        points_array = np.array(list(zip(ndcg_values, qps_values)))
        pareto_mask = is_pareto_efficient(points_array)
        pareto_points = points_array[pareto_mask]
        
        # Sort Pareto points by NDCG for line plot
        pareto_points = pareto_points[pareto_points[:, 0].argsort()]
        
        # Scatter plot for this querypool (all points or only Pareto points)
        if plot_all:
            scatter = ax.scatter(ndcg_values, qps_values, c=[color], label=f'querypool={querypool}')
        else:
            scatter = ax.scatter(pareto_points[:, 0], pareto_points[:, 1], c=[color], label=f'querypool={querypool}')
        
        # Plot Pareto frontier
        ax.plot(pareto_points[:, 0], pareto_points[:, 1], c=color, linestyle='--')
        
        # Create annotation texts
        for ndcg, qps, _, ann, candidates, _ in points:
            if plot_all or (ndcg, qps) in pareto_points:
                label = f"{querypool},{ann},{candidates}={ndcg:.3f}"
                texts.append(ax.text(ndcg, qps, label, fontsize=8))
    
    # Adjust text positions to minimize overlaps
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    ax.set_xlabel('NDCG@5')
    ax.set_ylabel('QPS')
    ax.set_title(f'{dataset_name}: NDCG@5 vs QPS')
    ax.legend()
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot NDCG@5 vs QPS for dataset results.")
    parser.add_argument("--dataset", help="Filter results to match this dataset name", default=None)
    parser.add_argument("--all", action="store_true", help="Plot all points, including non-Pareto-optimal ones")
    args = parser.parse_args()

    directory = 'outputs'
    data_points = read_data(directory, args.dataset)
    
    if not data_points:
        print(f"No data points found for dataset: {args.dataset}")
        return

    fig = plot_data(data_points, plot_all=args.all)
    
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

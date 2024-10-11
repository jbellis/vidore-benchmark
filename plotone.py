import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from adjustText import adjust_text
import argparse
import itertools
from typing import List, Tuple

def parse_filename(filename):
    parts = filename.split('_')
    candidates = int(parts[-1].split('.')[0])
    ann = float(parts[-2])
    querypool = float(parts[-3])
    docpool = int(parts[-4])
    dataset_name = '_'.join(parts[1:-4])  # Ignore the first part and last 4 parts
    return dataset_name, docpool, querypool, ann, candidates


def n_queries(dataset_name):
    if dataset_name.startswith('arxivqa'):
        return 50
    if dataset_name.startswith('docvqa'):
        return 45
    if dataset_name.startswith('infovqa'):
        return 49
    if dataset_name.startswith('tabfquad'):
        return 28
    if dataset_name.startswith('tatdqa'):
        return 164
    if dataset_name.startswith('shiftproject'):
        return 10
    if dataset_name.startswith('synthetic'):
        return 10
    raise ValueError(f"Unknown dataset: {dataset_name}")


def read_data(directory, dataset_filter=None):
    data_points = []
    for filename in os.listdir(directory):
        if filename.endswith('.pth') and not filename.startswith('all'):
            dataset_name, docpool, querypool, ann, candidates = parse_filename(filename)
            if dataset_filter and dataset_filter not in dataset_name:
                continue
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                ndcg_at_5 = data[next(iter(data))]['ndcg_at_5']
                elapsed = data[next(iter(data))]['elapsed']
                qps = n_queries(dataset_name) / elapsed
                data_points.append((ndcg_at_5, qps, docpool, querypool, ann, candidates, dataset_name))
    return data_points

def is_pareto_efficient(points):
    is_efficient = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] > point, axis=1)
            is_efficient[i] = True
    return is_efficient

def filter_tied_points(points: List[Tuple]) -> List[Tuple]:
    filtered_points = []
    points.sort(key=lambda x: (x[0], -x[1]))  # Sort by NDCG (ascending) then QPS (descending)
    
    for ndcg, group in itertools.groupby(points, key=lambda x: x[0]):
        group_list = list(group)
        best_point = group_list[0]  # First point is the one with highest QPS
        filtered_points.append(best_point)

    return filtered_points

def plot_data(data_points, plot_all=False, dataset_filter=None, groupby='docpool'):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data points by dataset if no dataset filter, otherwise by groupby option
    grouped_data = defaultdict(list)
    for point in data_points:
        if dataset_filter is None:
            key = point[6]  # Use dataset_name (index 6) when no filter
        elif groupby == 'docpool':
            key = point[2]  # Use docpool (index 2)
        elif groupby == 'querypool':
            key = point[3]  # Use querypool (index 3)
        else:
            raise ValueError(f"Invalid groupby option: {groupby}")
        grouped_data[key].append(point)
    
    # Generate a color map and marker styles
    color_map = plt.cm.viridis
    colors = [color_map(i) for i in np.linspace(0, 1, len(grouped_data))]
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H', '+', 'x'])
    
    texts = []
    for (group_key, points), color, marker in zip(grouped_data.items(), colors, markers):
        # Filter points
        filtered_points = filter_tied_points(points)
        
        ndcg_values, qps_values = zip(*[(p[0], p[1]) for p in filtered_points])
        
        # Find Pareto optimal points
        points_array = np.array(list(zip(ndcg_values, qps_values)))
        pareto_mask = is_pareto_efficient(points_array)
        pareto_points = points_array[pareto_mask]
        
        # Sort Pareto points by NDCG for line plot
        pareto_points = pareto_points[pareto_points[:, 0].argsort()]
        
        # Scatter plot (all points or only Pareto points)
        if plot_all:
            scatter = ax.scatter(ndcg_values, qps_values, c=[color], marker=marker, label=group_key)
        else:
            scatter = ax.scatter(pareto_points[:, 0], pareto_points[:, 1], c=[color], marker=marker, label=group_key)
        
        # Plot Pareto frontier
        ax.plot(pareto_points[:, 0], pareto_points[:, 1], c=color, linestyle='--')
        
        # Create annotation texts
        for ndcg, qps, docpool, querypool, ann, candidates, _ in filtered_points:
            if plot_all or (ndcg, qps) in pareto_points:
                label = f"{docpool},{querypool},{ann},{candidates}={ndcg:.3f}"
                texts.append(ax.text(ndcg, qps, label, fontsize=8))
    
    # Adjust text positions to minimize overlaps
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    ax.set_xlabel('NDCG@5')
    ax.set_ylabel('QPS')
    title = 'NDCG@5 vs QPS' if dataset_filter is None else f'{dataset_filter}: NDCG@5 vs QPS'
    ax.set_title(title)
    ax.legend()
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot NDCG@5 vs QPS for dataset results.")
    parser.add_argument("--dataset", help="Filter results to match this dataset name", default=None)
    parser.add_argument("--all", action="store_true", help="Plot all points, including non-Pareto-optimal ones")
    parser.add_argument("--groupby", choices=['docpool', 'querypool'], default='docpool',
                        help="Group results by docpool or querypool when using --dataset (default: docpool)")
    args = parser.parse_args()

    directory = 'outputs'
    data_points = read_data(directory, args.dataset)
    
    if not data_points:
        print(f"No data points found for dataset: {args.dataset}")
        return

    fig = plot_data(data_points, plot_all=args.all, dataset_filter=args.dataset, groupby=args.groupby)
    
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

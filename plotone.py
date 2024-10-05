import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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

def is_pareto_efficient(points):
    is_efficient = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] > point, axis=1)
            is_efficient[i] = True
    return is_efficient

def plot_data(data_points):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data points by querypool
    grouped_data = defaultdict(list)
    for point in data_points:
        grouped_data[point[2]].append(point)
    
    # Generate a color map
    color_map = plt.cm.viridis
    colors = [color_map(i) for i in np.linspace(0, 1, len(grouped_data))]
    
    for (querypool, points), color in zip(grouped_data.items(), colors):
        ndcg_values, qps_values = zip(*[(p[0], p[1]) for p in points])
        
        # Scatter plot for this querypool
        scatter = ax.scatter(ndcg_values, qps_values, c=[color], label=f'querypool={querypool}')
        
        # Find Pareto optimal points
        points_array = np.array(list(zip(ndcg_values, qps_values)))
        pareto_mask = is_pareto_efficient(points_array)
        pareto_points = points_array[pareto_mask]
        
        # Sort Pareto points by NDCG for line plot
        pareto_points = pareto_points[pareto_points[:, 0].argsort()]
        
        # Plot Pareto frontier
        ax.plot(pareto_points[:, 0], pareto_points[:, 1], c=color, linestyle='--')
        
        # Annotate points
        for ndcg, qps, _, ann, candidates, _ in points:
            label = f"{querypool},{ann},{candidates}"
            ax.annotate(label, (ndcg, qps), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('NDCG@5')
    ax.set_ylabel('QPS')
    ax.set_title('NDCG@5 vs QPS')
    ax.legend()
    
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

import os
import json
import re
from glob import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict

def parse_filename(filename):
    pattern = r'(.+?)_(\d+)_(\d+\.\d+)_(\d+)_(\d+)\.pth$'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        ignored, docpool, querypool, n_ann, n_maxsim = match.groups()
        if ignored == 'all':
            return None
        return {
            'dataset': ignored.split('_', 1)[1],
            'docpool': int(docpool),
            'querypool': float(querypool),
            'n_ann': int(n_ann),
            'n_maxsim': int(n_maxsim),
            'filename': filename
        }
    return None

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def process_group(group_data):
    X = np.array([[d['querypool'], d['n_ann'], d['n_maxsim']] for d in group_data])
    y = np.array([d['elapsed'] for d in group_data])
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    residuals = y - predictions
    std_dev = np.std(residuals)
    
    for i, d in enumerate(group_data):
        if residuals[i] > 1.0 * std_dev:
            print(d['filename'])
            # f"{d['filename']}: predicted {predictions[i]:.2f}, actual {d['elapsed']:.2f}")

def main():
    files = glob('outputs/*.pth')
    grouped_data = defaultdict(list)
    
    for file in files:
        parsed = parse_filename(file)
        if parsed:
            file_data = load_data(file)
            elapsed = list(file_data.values())[0]['elapsed']
            parsed['elapsed'] = elapsed
            group_key = (parsed['dataset'], parsed['docpool'])
            grouped_data[group_key].append(parsed)
    
    for (dataset, docpool), group_data in sorted(grouped_data.items(), key=lambda x: x[0][1]):
        if len(group_data) < 4:  # Minimum number of samples for regression
            print(f"Skipping group due to insufficient data (only {len(group_data)} samples)")
            continue
        process_group(group_data)

if __name__ == "__main__":
    main()

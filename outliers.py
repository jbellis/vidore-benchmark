import os
import json
import sys
import re
from glob import glob
import numpy as np
from sklearn.linear_model import LinearRegression

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

def main(dataset, docpool):
    files = glob('outputs/*.pth')
    data = []
    
    for file in files:
        parsed = parse_filename(file)
        if parsed and parsed['dataset'] == dataset and parsed['docpool'] == int(docpool):
            file_data = load_data(file)
            elapsed = list(file_data.values())[0]['elapsed']
            data.append({**parsed, 'elapsed': elapsed})
    
    if not data:
        print(f"No matching files found for dataset '{dataset}' and docpool '{docpool}'")
        return
    
    X = np.array([[d['querypool'], d['n_ann'], d['n_maxsim']] for d in data])
    y = np.array([d['elapsed'] for d in data])
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("Linear function coefficients:")
    print(f"f(querypool, n_ann, n_maxsim) = {model.intercept_:.4f} + "
          f"{model.coef_[0]:.4f} * querypool + "
          f"{model.coef_[1]:.4f} * n_ann + "
          f"{model.coef_[2]:.4f} * n_maxsim")
    
    predictions = model.predict(X)
    residuals = y - predictions
    std_dev = np.std(residuals)
    
    print("\nFiles where elapsed time is not within 2 standard deviations of the prediction:")
    for i, d in enumerate(data):
        if abs(residuals[i]) > 2 * std_dev:
            print(f"{d['filename']}: predicted {predictions[i]:.2f}, actual {d['elapsed']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <dataset> <docpool>")
        sys.exit(1)
    
    dataset = sys.argv[1]
    docpool = sys.argv[2]
    main(dataset, docpool)

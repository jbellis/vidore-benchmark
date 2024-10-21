import os
import json
import argparse
from collections import defaultdict

def parse_filename(filename):
    # Split off the prefix and suffix
    parts = filename.split('_')
    prefix = parts[0]
    suffix = parts[-1].split('.')[0]
    
    # Join the remaining parts as the dataset
    dataset = '_'.join(parts[1:-4])
    
    # Parse the remaining parameters
    docpool = int(parts[-4])
    querypool = float(parts[-3])
    ann = int(parts[-2])
    candidates = int(suffix)
    
    return {
        'dataset': dataset,
        'docpool': docpool,
        'querypool': querypool,
        'ann': ann,
        'candidates': candidates
    }

def process_directory(directory):
    results = defaultdict(list)
    
    for filename in os.listdir(directory):
        if filename.endswith('.pth') and filename.startswith('vidore'):
            file_path = os.path.join(directory, filename)
            params = parse_filename(filename)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Assuming the JSON structure is consistent
            ndcg_5 = list(data.values())[0]['ndcg_at_5']
            
            results[params['dataset']].append({
                'params': params,
                'ndcg_5': ndcg_5
            })
    
    return results

def find_best_params(results):
    best_params = {}
    
    for dataset, entries in results.items():
        best_entry = max(entries, key=lambda x: x['ndcg_5'])
        best_params[dataset] = {
            'params': best_entry['params'],
            'ndcg_5': best_entry['ndcg_5']
        }
    
    return best_params

def main():
    parser = argparse.ArgumentParser(description='Find best parameters for Vidore datasets')
    parser.add_argument('directory', type=str, help='Directory containing the .pth files')
    args = parser.parse_args()
    
    results = process_directory(args.directory)
    best_params = find_best_params(results)
    
    for dataset in sorted(best_params.keys()):
        entry = best_params[dataset]
        print(f"Dataset: {dataset}")
        print(f"Best ndcg@5: {entry['ndcg_5']:.5f}")
        print("Parameters:")
        for key, value in entry['params'].items():
            if key != 'dataset':
                print(f"  {key}: {value}")
        print()

if __name__ == "__main__":
    main()

import os
import json

def extract_dataset_and_model(filename):
    # Remove 'vidore_' prefix and '.pth' suffix
    content = filename.replace('vidore_', '').replace('.pth', '')
    
    # Hardcode model names
    models = ['bm25', 'bge_m3']
    
    # Define OCR types
    ocr_types = ['unstructured', 'flash', 'llamaparse']
    
    # Find model in filename
    model = next((m for m in models if m in content), None)
    
    if not model:
        return None, None, None
    
    # Find OCR type in filename
    ocr_type = next((o for o in ocr_types if o in content), None)
    
    if not ocr_type:
        return None, None, None
    
    # Extract dataset name
    dataset = content.split(f"{ocr_type}_{model}")[0].rstrip('_')
    
    return dataset, ocr_type, model

def read_ndcg_value(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]['ndcg_at_5']

def main():
    output_dir = 'outputs'
    results = {}

    for filename in os.listdir(output_dir):
        if filename.startswith('vidore_') and filename.endswith('.pth'):
            file_path = os.path.join(output_dir, filename)
            dataset, ocr_type, model = extract_dataset_and_model(filename)
            if dataset and ocr_type and model:
                ndcg_value = read_ndcg_value(file_path)
                
                if dataset not in results:
                    results[dataset] = {}
                results[dataset][f"{ocr_type}_{model}"] = ndcg_value
            else:
                print(f"Warning: Unable to parse dataset, OCR type, and model from file '{filename}'. Skipping this file.")

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

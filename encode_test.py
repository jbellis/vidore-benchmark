import torch
import json
import os
import sys
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
import numpy as np

def main(doc_ordinal: int):
    DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
    model_path = "/home/jonathan/datasets/arxivqa/fine_tuned_stella_en_400M_v5_None_6400"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model first
    model = AutoModel.from_pretrained(
        "dunzhang/stella_en_400M_v5",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Then load our fine-tuned weights
    state_dict = load_file(f"{model_path}/model.safetensors")
    
    # Remove 'base_model.' prefix from state dict keys if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('base_model.'):
            cleaned_state_dict[k[len('base_model.'):]] = v
        else:
            cleaned_state_dict[k] = v
            
    # Load the cleaned state dict
    model.load_state_dict(cleaned_state_dict)
    
    # Move to device and set dtype
    model = model.to(dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Load the document and its corresponding query
    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')
    
    # Get all OCR files
    all_files = sorted(os.listdir(ocr_dir))
    if doc_ordinal >= len(all_files):
        print(f"Error: doc_ordinal {doc_ordinal} exceeds number of files {len(all_files)}")
        sys.exit(1)
        
    filename = all_files[doc_ordinal]
    file_hash = os.path.splitext(filename)[0]
    
    # Read the OCR text
    with open(os.path.join(ocr_dir, filename), 'r') as f:
        doc_text = f.read().strip()
        
    # Find corresponding question
    query_text = None
    with open(preprocessed_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['image_hash'] == file_hash:
                query_text = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " + data['question']
                break
    
    if query_text is None:
        print(f"Error: No query found for document hash {file_hash}")
        sys.exit(1)
        
    print(f"Document: {doc_text[:100]}...")
    print(f"Query: {query_text}")
    
    # Encode both texts
    with torch.no_grad():
        # Encode document
        doc_inputs = tokenizer(
            doc_text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        doc_outputs = model(**doc_inputs)
        doc_embedding = doc_outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
        
        # Encode query
        query_inputs = tokenizer(
            query_text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        query_outputs = model(**query_inputs)
        query_embedding = query_outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
        
    # Calculate cosine similarity
    similarity = np.dot(doc_embedding[0], query_embedding[0]) / (
        np.linalg.norm(doc_embedding[0]) * np.linalg.norm(query_embedding[0])
    )
    
    print(f"\nCosine similarity between document and query: {similarity:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encode_test.py DOC_ORDINAL")
        sys.exit(1)
    main(int(sys.argv[1]))

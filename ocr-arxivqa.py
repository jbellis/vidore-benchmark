import json
import os
from os import path
from PIL import Image
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.vidore_benchmark.retrievers.colbert_live_retriever import encode_to_bytes
from src.vidore_benchmark.retrievers.ocr_providers import GeminiOcrProvider

DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
TEST_LOCATION = "document_cache_flash/vidore/arxivqa_test_subsampled"


def process_arxivqa_line(data, ocr_provider, test_files):
    image_path = path.join(DATASET_LOCATION, data['image'])
    
    # Open and process the image
    with Image.open(image_path) as img:
        # Use the precomputed hash from the preprocessed data
        sha256_hash = data['image_hash']
        
        # Skip if the hash is in test_files
        if f"{sha256_hash}.txt" in test_files:
            return
        
        # Perform OCR
        ocr_text = ocr_provider.ocr(img, sha256_hash)
        
        # Write OCR text to file in the OCR directory
        ocr_dir = path.join(DATASET_LOCATION, 'ocr')
        os.makedirs(ocr_dir, exist_ok=True)
        with open(path.join(ocr_dir, f"{sha256_hash}.txt"), "w") as f:
            f.write(ocr_text)


def main():
    ocr_provider = GeminiOcrProvider()
    test_files = set(os.listdir(TEST_LOCATION))
    
    with open(output_file, 'r') as file:
        for line in tqdm(file, desc="Processing ArXivQA lines"):
            data = json.loads(line.strip())
            process_arxivqa_line(data, ocr_provider, test_files)

if __name__ == "__main__":
    main()

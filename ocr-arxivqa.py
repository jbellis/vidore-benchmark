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

def preprocess(input_file, output_file):
    def process_line(line):
        data = json.loads(line)
        image_path = path.join(DATASET_LOCATION, data['image'])

        with Image.open(image_path) as img:
            img_bytes = encode_to_bytes(img)
            sha256_hash = hashlib.sha256(img_bytes).hexdigest()

        return {
            'image': data['image'],
            'question': data['question'],
            'image_hash': sha256_hash
        }

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_line, lines), total=len(lines), desc="Preprocessing"))

        for result in results:
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write('\n')


def process_arxivqa_line(data, ocr_provider, test_files):
    image_path = path.join(DATASET_LOCATION, data['image'])
    sha256_hash = data['image_hash']
    ocr_dir = path.join(DATASET_LOCATION, 'ocr')
    ocr_file_path = path.join(ocr_dir, f"{sha256_hash}.txt")

    # Skip if the hash is in test_files or if the OCR file already exists
    if f"{sha256_hash}.txt" in test_files or path.exists(ocr_file_path):
        return

    # Open and process the image
    with Image.open(image_path) as img:
        # Perform OCR
        ocr_text = ocr_provider.ocr(img, sha256_hash)
        
        # Write OCR text to file in the OCR directory
        with open(ocr_file_path, "w") as f:
            f.write(ocr_text)


def main():
    raw_file = os.path.join(DATASET_LOCATION, 'arxivqa.jsonl')
    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    # preprocess(raw_file, preprocessed_file)
    # return

    ocr_provider = GeminiOcrProvider()
    test_files = set(os.listdir(TEST_LOCATION))
    ocr_dir = path.join(DATASET_LOCATION, 'ocr')
    os.makedirs(ocr_dir, exist_ok=True)

    with open(preprocessed_file, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing ArXivQA images"):
            data = json.loads(line.strip())
            process_arxivqa_line(data, ocr_provider, test_files)

if __name__ == "__main__":
    main()

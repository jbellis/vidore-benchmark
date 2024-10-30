import os
from os import path

from PIL import Image
from tqdm import tqdm

from src.vidore_benchmark.retrievers.ocr_providers import GeminiOcrProvider

DATASET_LOCATION = "/home/jonathan/datasets/infovqa"

def ocr_one_file(image_path, ocr_file_path, ocr_provider):
    # Open and process the image
    with Image.open(image_path) as img:
        # Perform OCR
        ocr_text = ocr_provider.ocr(img, image_path)
        if ocr_text is None:
            print('No text extracted for ', image_path)
            return

        # Write OCR text to file in the OCR directory
        with open(ocr_file_path, "w") as f:
            f.write(ocr_text)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, help="Stop after processing N files successfully")
    args = parser.parse_args()

    ocr_provider = GeminiOcrProvider()
    jpeg_dir = os.path.join(DATASET_LOCATION, 'jpeg')
    jpeg_files = set(os.listdir(jpeg_dir))

    ocr_dir = path.join(DATASET_LOCATION, 'ocr')
    os.makedirs(ocr_dir, exist_ok=True)

    processed_count = 0
    for fname in tqdm(jpeg_files, desc="Processing InfoVQA images"):
        input_path = path.join(jpeg_dir, fname)
        output_path = path.join(ocr_dir, fname)

        if path.exists(output_path):
            continue

        ocr_one_file(input_path, output_path, ocr_provider)

        # Only increment counter if OCR file exists (meaning OCR succeeded)
        if path.exists(output_path):
            processed_count += 1
            if args.files and processed_count >= args.files:
                print(f"\nStopping after {processed_count} successful OCR files")
                break

if __name__ == "__main__":
    main()

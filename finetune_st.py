import argparse
import json
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import Dataset, DataLoader

DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
SEQUENCE_LENGTH = 512

class ArxivQADataset(Dataset):
    def __init__(self, preprocessed_file: str, ocr_dir: str, start_file: int, end_file: int):
        self.queries = []
        self.texts = []
        self.load_data(preprocessed_file, ocr_dir, start_file, end_file)

    def load_data(self, preprocessed_file: str, ocr_dir: str, start_file: int, end_file: int):
        hash_to_question = {}
        with open(preprocessed_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                hash_to_question[data['image_hash']] = data['question']

        all_files = sorted(os.listdir(ocr_dir))
        for i, filename in enumerate(all_files[start_file:end_file], start=start_file):
            file_hash = os.path.splitext(filename)[0]
            with open(os.path.join(ocr_dir, filename), 'r') as f:
                ocr_text = f.read().strip()
            
            if file_hash in hash_to_question:
                self.queries.append(hash_to_question[file_hash])
                self.texts.append(ocr_text)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return InputExample(
            texts=[self.queries[idx], self.texts[idx]]
        )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SentenceTransformer model on ArxivQA dataset")
    parser.add_argument("--train-files", type=int, default=1000, help="Number of files to use for training")
    parser.add_argument("--val-files", type=int, default=100, help="Number of files to use for validation")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-large-en-v1.5", help="Model to fine-tune")
    parser.add_argument("--output-dir", type=str, help="Directory to save model checkpoints")
    args = parser.parse_args()

    # Set default output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(DATASET_LOCATION, f"st-checkpoints-{timestamp}")

    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')

    # Initialize the model
    model = SentenceTransformer(args.model, trust_remote_code=True)

    # Create datasets and dataloaders
    train_dataset = ArxivQADataset(preprocessed_file, ocr_dir, 0, args.train_files)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_dataset = ArxivQADataset(
        preprocessed_file,
        ocr_dir,
        args.train_files,
        args.train_files + args.val_files
    )

    # Create the training loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Create evaluator
    evaluator = InformationRetrievalEvaluator(
        queries={str(i): q for i, q in enumerate(val_dataset.queries)},
        corpus={str(i): t for i, t in enumerate(val_dataset.texts)},
        relevant_docs={str(i): {str(i)} for i in range(len(val_dataset))},
        show_progress_bar=True
    )

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=100,
        output_path=args.output_dir,
        show_progress_bar=True
    )

if __name__ == "__main__":
    main()

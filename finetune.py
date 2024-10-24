import argparse
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import losses, SentenceTransformer
from tqdm import tqdm

DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate 4x larger batch size

class ArxivQADataset(Dataset):
    def __init__(self, preprocessed_file: str, ocr_dir: str, num_files: int, tokenizer):
        self.questions = []
        self.ocr_texts = []
        self.tokenizer = tokenizer
        self.load_data(preprocessed_file, ocr_dir, num_files)

    def load_data(self, preprocessed_file: str, ocr_dir: str, num_files: int):
        hash_to_question = {}
        with open(preprocessed_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                hash_to_question[data['image_hash']] = data['question']

        for i, filename in enumerate(os.listdir(ocr_dir)):
            if i >= num_files:
                break
            
            file_hash = os.path.splitext(filename)[0]
            if file_hash not in hash_to_question:
                continue

            with open(os.path.join(ocr_dir, filename), 'r') as f:
                ocr_text = f.read().strip()

            question = hash_to_question[file_hash]
            self.questions.append(question)
            self.ocr_texts.append(ocr_text)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        ocr_text = self.ocr_texts[idx]
        
        question_encoding = self.tokenizer(question, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        ocr_encoding = self.tokenizer(ocr_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        
        return {
            'question_input_ids': question_encoding['input_ids'].squeeze(0),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(0),
            'ocr_input_ids': ocr_encoding['input_ids'].squeeze(0),
            'ocr_attention_mask': ocr_encoding['attention_mask'].squeeze(0),
        }

def train(model, train_dataloader, epochs: int, device: str):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            question_features = {
                'input_ids': batch['question_input_ids'].to(device),
                'attention_mask': batch['question_attention_mask'].to(device)
            }
            ocr_features = {
                'input_ids': batch['ocr_input_ids'].to(device),
                'attention_mask': batch['ocr_attention_mask'].to(device)
            }

            print(f"Batch size: {len(batch['question_input_ids'])}")
            print(f"Question input_ids shape: {question_features['input_ids'].shape}")
            print(f"Question attention_mask shape: {question_features['attention_mask'].shape}")
            print(f"OCR input_ids shape: {ocr_features['input_ids'].shape}")
            print(f"OCR attention_mask shape: {ocr_features['attention_mask'].shape}")

            # Compute embeddings
            question_embeddings = model(question_features)['sentence_embedding']
            ocr_embeddings = model(ocr_features)['sentence_embedding']

            # Combine embeddings
            embeddings = torch.cat([question_embeddings, ocr_embeddings])

            # Create labels (0 for questions, 1 for OCR texts)
            labels = torch.arange(question_embeddings.size(0), device=device)

            # Compute loss
            loss = loss_fn(embeddings, labels)
            
            total_loss += loss.item()
            loss = loss / GRADIENT_ACCUMULATION_STEPS  # Normalize the loss
            loss.backward()
            
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GTE-large embeddings model on ArxivQA dataset")
    parser.add_argument("--num-files", required=True, type=int, help="Number of files to use for training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()

    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')

    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5')
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    dataset = ArxivQADataset(preprocessed_file, ocr_dir, args.num_files, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train(model, dataloader, args.epochs, args.device)

    output_path = os.path.join(DATASET_LOCATION, 'fine_tuned_gte_large')
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Fine-tuned model saved to {output_path}")

if __name__ == "__main__":
    main()

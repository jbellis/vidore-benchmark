import argparse
import json
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate 4x larger batch size
SEQUENCE_LENGTH = 512

class ArxivQADataset(Dataset):
    def __init__(self, preprocessed_file: str, ocr_dir: str, num_files: int, tokenizer):
        self.questions = []
        self.positive_ocr_texts = []
        self.all_ocr_texts = []
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
            with open(os.path.join(ocr_dir, filename), 'r') as f:
                ocr_text = f.read().strip()
            self.all_ocr_texts.append(ocr_text)

            if file_hash in hash_to_question:
                question = hash_to_question[file_hash]
                self.questions.append(question)
                self.positive_ocr_texts.append(ocr_text)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        positive_ocr_text = self.positive_ocr_texts[idx]
        
        # Randomly select a negative example that's different from the positive one
        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.all_ocr_texts) - 1)
        negative_ocr_text = self.all_ocr_texts[negative_idx]
        
        question_encoding = self.tokenizer(question, truncation=True, padding='max_length', max_length=SEQUENCE_LENGTH, return_tensors='pt')
        positive_ocr_encoding = self.tokenizer(positive_ocr_text, truncation=True, padding='max_length', max_length=SEQUENCE_LENGTH, return_tensors='pt')
        negative_ocr_encoding = self.tokenizer(negative_ocr_text, truncation=True, padding='max_length', max_length=SEQUENCE_LENGTH, return_tensors='pt')
        
        return {
            'question_input_ids': question_encoding['input_ids'].squeeze(),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(),
            'positive_ocr_input_ids': positive_ocr_encoding['input_ids'].squeeze(),
            'positive_ocr_attention_mask': positive_ocr_encoding['attention_mask'].squeeze(),
            'negative_ocr_input_ids': negative_ocr_encoding['input_ids'].squeeze(),
            'negative_ocr_attention_mask': negative_ocr_encoding['attention_mask'].squeeze(),
        }


def train(model, train_dataloader, epochs: int, device: str):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    # Calculate initial loss
    model.eval()
    initial_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Calculating initial loss"):
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            positive_ocr_input_ids = batch['positive_ocr_input_ids'].to(device)
            positive_ocr_attention_mask = batch['positive_ocr_attention_mask'].to(device)
            negative_ocr_input_ids = batch['negative_ocr_input_ids'].to(device)
            negative_ocr_attention_mask = batch['negative_ocr_attention_mask'].to(device)

            question_embeddings = model(input_ids=question_input_ids, attention_mask=question_attention_mask).last_hidden_state[:, 0, :]
            positive_ocr_embeddings = model(input_ids=positive_ocr_input_ids, attention_mask=positive_ocr_attention_mask).last_hidden_state[:, 0, :]
            negative_ocr_embeddings = model(input_ids=negative_ocr_input_ids, attention_mask=negative_ocr_attention_mask).last_hidden_state[:, 0, :]

            loss = loss_fn(question_embeddings, positive_ocr_embeddings, negative_ocr_embeddings)
            initial_loss += loss.item()

    initial_avg_loss = initial_loss / len(train_dataloader)
    print(f"Initial Average Loss: {initial_avg_loss:.4f}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            positive_ocr_input_ids = batch['positive_ocr_input_ids'].to(device)
            positive_ocr_attention_mask = batch['positive_ocr_attention_mask'].to(device)
            negative_ocr_input_ids = batch['negative_ocr_input_ids'].to(device)
            negative_ocr_attention_mask = batch['negative_ocr_attention_mask'].to(device)

            question_embeddings = model(input_ids=question_input_ids, attention_mask=question_attention_mask).last_hidden_state[:, 0, :]
            positive_ocr_embeddings = model(input_ids=positive_ocr_input_ids, attention_mask=positive_ocr_attention_mask).last_hidden_state[:, 0, :]
            negative_ocr_embeddings = model(input_ids=negative_ocr_input_ids, attention_mask=negative_ocr_attention_mask).last_hidden_state[:, 0, :]

            loss = loss_fn(question_embeddings, positive_ocr_embeddings, negative_ocr_embeddings)
            
            loss = loss / GRADIENT_ACCUMULATION_STEPS  # Normalize the loss
            loss.backward()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GTE-large embeddings model on ArxivQA dataset")
    parser.add_argument("--num-files", type=int, default=1000, help="Number of files to use for training")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()

    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')

    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model.to(args.device)  # Move the model to the specified device

    dataset = ArxivQADataset(preprocessed_file, ocr_dir, args.num_files, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train the model
    train(model, dataloader, args.epochs, args.device)

    # Save the fine-tuned model
    output_path = os.path.join(DATASET_LOCATION, f'fine_tuned_gte_large_{args.num_files}')
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Fine-tuned model saved to {output_path}")

if __name__ == "__main__":
    main()

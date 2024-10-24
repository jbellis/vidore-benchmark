import argparse
import json
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
GRADIENT_ACCUMULATION_STEPS = 8  # Simulate 4x larger batch size
SEQUENCE_LENGTH = 512

class ArxivQADataset(Dataset):
    def __init__(self, preprocessed_file: str, ocr_dir: str, start_file: int, end_file: int, tokenizer):
        self.questions = []
        self.positive_ocr_texts = []
        self.all_ocr_texts = []
        self.tokenizer = tokenizer
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


def train(model, train_dataloader, val_dataloader, device: str, patience: int = 3, max_epochs: int = 100, skip_validations: int = 3):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast():
                question_embeddings = model(input_ids=batch['question_input_ids'], attention_mask=batch['question_attention_mask']).last_hidden_state[:, 0, :]
                positive_ocr_embeddings = model(input_ids=batch['positive_ocr_input_ids'], attention_mask=batch['positive_ocr_attention_mask']).last_hidden_state[:, 0, :]
                negative_ocr_embeddings = model(input_ids=batch['negative_ocr_input_ids'], attention_mask=batch['negative_ocr_attention_mask']).last_hidden_state[:, 0, :]
                loss = loss_fn(question_embeddings, positive_ocr_embeddings, negative_ocr_embeddings)
            
            scaler.scale(loss).backward()
            
            total_loss += loss.item()
            
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        
        if epoch >= skip_validations:
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    question_embeddings = model(input_ids=batch['question_input_ids'], attention_mask=batch['question_attention_mask']).last_hidden_state[:, 0, :]
                    positive_ocr_embeddings = model(input_ids=batch['positive_ocr_input_ids'], attention_mask=batch['positive_ocr_attention_mask']).last_hidden_state[:, 0, :]
                    negative_ocr_embeddings = model(input_ids=batch['negative_ocr_input_ids'], attention_mask=batch['negative_ocr_attention_mask']).last_hidden_state[:, 0, :]
                    
                    loss = loss_fn(question_embeddings, positive_ocr_embeddings, negative_ocr_embeddings)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            
            print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        else:
            print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Validation skipped")
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GTE-large embeddings model on ArxivQA dataset")
    parser.add_argument("--train-files", type=int, default=1000, help="Number of files to use for training")
    parser.add_argument("--val-files", type=int, default=100, help="Number of files to use for validation")
    parser.add_argument("--batch-size", type=int, default=7, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--skip-validations", type=int, default=3, help="Number of initial epochs to skip validation")
    args = parser.parse_args()

    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')

    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    train_dataset = ArxivQADataset(preprocessed_file, ocr_dir, 0, args.train_files, tokenizer)
    val_dataset = ArxivQADataset(preprocessed_file, ocr_dir, args.train_files, args.train_files + args.val_files, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Train the model
    model = train(model, train_dataloader, val_dataloader, args.device, args.patience, args.max_epochs, args.skip_validations)

    # Save the fine-tuned model
    output_path = os.path.join(DATASET_LOCATION, f'fine_tuned_gte_large_{args.train_files}')
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Fine-tuned model saved to {output_path}")

if __name__ == "__main__":
    main()

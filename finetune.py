import argparse
import json
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

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

        question_encoding = self.tokenizer(question, truncation=True, padding='max_length', max_length=SEQUENCE_LENGTH,
                                           return_tensors='pt')
        positive_ocr_encoding = self.tokenizer(positive_ocr_text, truncation=True, padding='max_length',
                                               max_length=SEQUENCE_LENGTH, return_tensors='pt')
        negative_ocr_encoding = self.tokenizer(negative_ocr_text, truncation=True, padding='max_length',
                                               max_length=SEQUENCE_LENGTH, return_tensors='pt')

        # Format expected by Trainer
        return {
            'input_ids': question_encoding['input_ids'].squeeze(),
            'attention_mask': question_encoding['attention_mask'].squeeze(),
            'positive_ids': positive_ocr_encoding['input_ids'].squeeze(),
            'positive_mask': positive_ocr_encoding['attention_mask'].squeeze(),
            'negative_ids': negative_ocr_encoding['input_ids'].squeeze(),
            'negative_mask': negative_ocr_encoding['attention_mask'].squeeze(),
        }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GTE-large embeddings model on ArxivQA dataset")
    parser.add_argument("--train-files", type=int, default=1000, help="Number of files to use for training")
    parser.add_argument("--val-files", type=int, default=100, help="Number of files to use for validation")
    parser.add_argument("--batch-size", type=int, default=7, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-large-en-v1.5", help="Model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    class TripletModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

        def forward(self, input_ids, attention_mask, positive_ids, positive_mask, negative_ids, negative_mask):
            # Get embeddings for each input
            query_emb = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            positive_emb = self.base_model(input_ids=positive_ids, attention_mask=positive_mask).last_hidden_state[:, 0, :]
            negative_emb = self.base_model(input_ids=negative_ids, attention_mask=negative_mask).last_hidden_state[:, 0, :]
            
            # Compute triplet loss
            loss = self.loss_fn(query_emb, positive_emb, negative_emb)
            
            return {"loss": loss, "logits": query_emb}

        def get_embedding(self, input_ids, attention_mask):
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model = TripletModel(base_model)
    
    train_dataset = ArxivQADataset(preprocessed_file, ocr_dir, 0, args.train_files, tokenizer)
    val_dataset = ArxivQADataset(preprocessed_file, ocr_dir, args.train_files, args.train_files + args.val_files,
                                 tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir='./training-logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        label_names = ["positive_ids", "positive_mask", "negative_ids", "negative_mask"]
    )

    class TripletCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.pad_token_id = tokenizer.pad_token_id

        def __call__(self, features):
            batch = {}
            
            # Pad and create tensor for each key
            for key in ['input_ids', 'attention_mask', 'positive_ids', 'positive_mask', 'negative_ids', 'negative_mask']:
                if key in features[0]:
                    batch[key] = torch.nn.utils.rnn.pad_sequence(
                        [f[key] for f in features],
                        batch_first=True,
                        padding_value=self.pad_token_id if 'ids' in key else 0
                    )
            
            return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=TripletCollator(tokenizer),
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model_name = args.model.split('/')[-1]
    output_path = os.path.join(DATASET_LOCATION, f'fine_tuned_{model_name}_{args.train_files}')
    model.base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Fine-tuned model saved to {output_path}")


if __name__ == "__main__":
    main()

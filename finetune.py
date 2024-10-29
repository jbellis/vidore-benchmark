import argparse
import json
import os
import random
import torch
from torch.utils.data import Dataset
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

torch.set_float32_matmul_precision('medium')


DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
SEQUENCE_LENGTH = 512


class ArxivQADataset(Dataset):
    def __init__(self, preprocessed_file: str, ocr_dir: str, start_file: int, end_file: int, tokenizer):
        self.tokenizer = tokenizer
        self.encoded_questions = []
        self.encoded_positive_texts = []
        self.encoded_all_texts = []
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
            # Pre-tokenize the OCR text
            encoded_ocr = self.tokenizer(
                ocr_text,
                truncation=True,
                padding='max_length',
                max_length=SEQUENCE_LENGTH,
                return_tensors='pt'
            )
            self.encoded_all_texts.append({
                'input_ids': encoded_ocr['input_ids'][0],
                'attention_mask': encoded_ocr['attention_mask'][0]
            })

            if file_hash in hash_to_question:
                question = hash_to_question[file_hash]
                # Pre-tokenize the question
                encoded_q = self.tokenizer(
                    question,
                    truncation=True,
                    padding='max_length',
                    max_length=SEQUENCE_LENGTH,
                    return_tensors='pt'
                )
                self.encoded_questions.append({
                    'input_ids': encoded_q['input_ids'][0],
                    'attention_mask': encoded_q['attention_mask'][0]
                })
                self.encoded_positive_texts.append({
                    'input_ids': encoded_ocr['input_ids'][0],
                    'attention_mask': encoded_ocr['attention_mask'][0]
                })

    def __len__(self):
        return len(self.encoded_questions)

    def __getitem__(self, idx):
        # Get pre-tokenized question and positive text
        question = self.encoded_questions[idx]
        positive = self.encoded_positive_texts[idx]

        # Randomly select a negative example
        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.encoded_all_texts) - 1)
        negative = self.encoded_all_texts[negative_idx]

        return {
            'input_ids': question['input_ids'],
            'attention_mask': question['attention_mask'],
            'positive_ids': positive['input_ids'],
            'positive_mask': positive['attention_mask'],
            'negative_ids': negative['input_ids'],
            'negative_mask': negative['attention_mask'],
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
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--gradient", type=int, default=8, help="Gradient accumulation steps to simulate larger batch size")
    parser.add_argument("--output-dim", type=int, default=1024, help="Output dimension of the embeddings")
    args = parser.parse_args()

    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    class TripletModel(torch.nn.Module):
        def __init__(self, base_model, output_dim):
            super().__init__()
            self.base_model = base_model
            self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0)
            # Add projection layer only if we need dimension reduction
            self.output_dim = output_dim
            if output_dim != self.base_model.config.hidden_size:
                self.projection = torch.nn.Linear(self.base_model.config.hidden_size, output_dim)
            else:
                self.projection = None

        def forward(self, input_ids, attention_mask, positive_ids, positive_mask, negative_ids, negative_mask):
            # Get embeddings for each input
            query_emb = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            positive_emb = self.base_model(input_ids=positive_ids, attention_mask=positive_mask).last_hidden_state[:, 0, :]
            negative_emb = self.base_model(input_ids=negative_ids, attention_mask=negative_mask).last_hidden_state[:, 0, :]
            
            if self.projection is not None:
                query_emb = self.projection(query_emb)
                positive_emb = self.projection(positive_emb)
                negative_emb = self.projection(negative_emb)
            
            # L2 normalize all embeddings
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1)
            negative_emb = torch.nn.functional.normalize(negative_emb, p=2, dim=1)
            
            # Compute triplet loss
            loss = self.loss_fn(query_emb, positive_emb, negative_emb)

            return {"loss": loss, "logits": query_emb}

        def gradient_checkpointing_enable(self, **kwargs):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model = TripletModel(base_model, args.output_dim)

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
        warmup_ratio=0.1,
        logging_dir='./training-logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=args.gradient,
        label_names=["positive_ids", "positive_mask", "negative_ids", "negative_mask"],
        gradient_checkpointing=True,  # Save memory
        dataloader_pin_memory=False,
    )
    class TripletCollator:
        def __init__(self, tokenizer, max_batch_size=32):
            self.tokenizer = tokenizer
            self.pad_token_id = tokenizer.pad_token_id
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.max_batch_size = max_batch_size
            self.max_len = SEQUENCE_LENGTH

            # Preallocate tensors on device
            self.batch_tensors = {
                'input_ids': torch.zeros(max_batch_size, SEQUENCE_LENGTH, dtype=torch.long, device=self.device),
                'attention_mask': torch.zeros(max_batch_size, SEQUENCE_LENGTH, dtype=torch.long, device=self.device),
                'positive_ids': torch.zeros(max_batch_size, SEQUENCE_LENGTH, dtype=torch.long, device=self.device),
                'positive_mask': torch.zeros(max_batch_size, SEQUENCE_LENGTH, dtype=torch.long, device=self.device),
                'negative_ids': torch.zeros(max_batch_size, SEQUENCE_LENGTH, dtype=torch.long, device=self.device),
                'negative_mask': torch.zeros(max_batch_size, SEQUENCE_LENGTH, dtype=torch.long, device=self.device)
            }

        def __call__(self, features):
            batch_size = len(features)

            # Reset tensors
            for tensor in self.batch_tensors.values():
                tensor.zero_()

            # Fill preallocated tensors
            for i, feature in enumerate(features):
                for key in self.batch_tensors:
                    seq_len = feature[key].size(0)
                    self.batch_tensors[key][i, :seq_len] = feature[key].to(self.device, non_blocking=True)

            # Return views of the actual batch size
            return {k: v[:batch_size] for k, v in self.batch_tensors.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=TripletCollator(tokenizer, max_batch_size=args.batch_size),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # Train the model
    trainer.train()
    
    # Save the fine-tuned model and projection layer
    model_name = args.model.split('/')[-1]
    output_path = os.path.join(DATASET_LOCATION, f'fine_tuned_{model_name}_{args.output_dim}_{args.train_files}')
    
    # Save base model and tokenizer
    model.base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save projection layer if it exists
    if model.projection is not None:
        projection_state = {
            'projection': model.projection.state_dict(),
            'output_dim': args.output_dim
        }
        torch.save(projection_state, os.path.join(output_path, 'projection_layer.pt'))
    
    print(f"Fine-tuned model and projection layer saved to {output_path}")


if __name__ == "__main__":
    main()

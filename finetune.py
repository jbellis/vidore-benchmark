import argparse
import json
import os
import random
import torch
from torch.utils.data import Dataset
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
    def __init__(self, preprocessed_file: str, ocr_dir: str, start_file: int, end_file: int, tokenizer, model_name: str):
        self.tokenizer = tokenizer
        self.encoded_questions = []
        self.encoded_positive_texts = []
        self.encoded_all_texts = []
        if 'stella' in model_name.lower():
            self.prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {question}"
        else:
            self.prompt = None
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
                if self.prompt is not None:
                    question = self.prompt.format(question=question)
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
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-large-en-v1.5", help="Model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--output-dim", type=int, help="Output dimension of the embeddings")
    args = parser.parse_args()

    # Calculate gradient accumulation steps: 128/batch_size, clamped between 1 and 64
    gradient_accumulation_steps = min(max(128 // args.batch_size, 1), 64)
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    args.learning_rate = 2e-5 * effective_batch_size / 64
    print(f"Using LR {args.learning_rate} with {gradient_accumulation_steps} steps")

    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    class TripletModel(torch.nn.Module):
        def __init__(self, base_model, output_dim):
            super().__init__()
            self.base_model = base_model
            self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.3)
            # Add projection layer only if we need dimension reduction
            print('Native encoding dimension is', self.base_model.config.hidden_size)
            if output_dim is None:
                self.projection = None
            else:
                print('Adding projection layer to', output_dim)
                self.projection = torch.nn.Linear(self.base_model.config.hidden_size, output_dim)

        def forward(self, input_ids, attention_mask, positive_ids, positive_mask, negative_ids, negative_mask):
            # Get embeddings for each input
            query_out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            query_emb = query_out.last_hidden_state[:, 0, :]

            positive_out = self.base_model(input_ids=positive_ids, attention_mask=positive_mask)
            positive_emb = positive_out.last_hidden_state[:, 0, :]

            negative_out = self.base_model(input_ids=negative_ids, attention_mask=negative_mask)
            negative_emb = negative_out.last_hidden_state[:, 0, :]
            
            if self.projection is not None:
                query_emb = self.projection(query_emb)
                positive_emb = self.projection(positive_emb)
                negative_emb = self.projection(negative_emb)
                # L2 normalize all embeddings with small epsilon for stability
                eps = 1e-8
                query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1, eps=eps)
                positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1, eps=eps)
                negative_emb = torch.nn.functional.normalize(negative_emb, p=2, dim=1, eps=eps)

            # CosineEmbeddingLoss uses the target tensor to determine the objective (pos or neg)
            pos_target = torch.ones(query_emb.size(0), device=query_emb.device)
            neg_target = -torch.ones(query_emb.size(0), device=query_emb.device)
            
            pos_loss = self.loss_fn(query_emb, positive_emb, pos_target)
            neg_loss = self.loss_fn(query_emb, negative_emb, neg_target)
            loss = pos_loss + neg_loss

            return {"loss": loss, "logits": query_emb}

        def gradient_checkpointing_enable(self, **kwargs):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    try:
        base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    except ValueError as e:
        if "does not support Flash Attention" in str(e):
            print('Flash attention not supported for', args.model, '; falling back to default attn')
            base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
        else:
            raise
    base_model = base_model.to('cuda')
    model = TripletModel(base_model, args.output_dim)

    train_dataset = ArxivQADataset(preprocessed_file, ocr_dir, 0, args.train_files, tokenizer, args.model)
    val_dataset = ArxivQADataset(preprocessed_file, ocr_dir, args.train_files, args.train_files + args.val_files,
                                 tokenizer, args.model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir='./training-logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        label_names=["positive_ids", "positive_mask", "negative_ids", "negative_mask"],
        gradient_checkpointing=True,  # Save memory
        gradient_checkpointing_kwargs={"use_reentrant": False},  # More stable checkpointing
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
    print(f"Fine-tuned model saved to {output_path}")

    # Save projection layer if it exists
    if model.projection is not None:
        projection_state = {
            'projection': model.projection.state_dict(),
            'output_dim': args.output_dim
        }
        projection_path = os.path.join(output_path, 'projection_layer.pt')
        torch.save(projection_state, projection_path)
        print(f"Projection layer saved to {projection_path}")

if __name__ == "__main__":
    main()

import argparse
from datetime import datetime
import json
import os
import random
import torch
from torch.utils.data import Dataset
import copy
import threading
import shutil
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
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
        if 'stella' in model_name or 'Qwen2' in model_name:
            print('Using prompt prefix when encoding queries for', model_name)
            self.prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        else:
            self.prompt = ""
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
                question = self.prompt + hash_to_question[file_hash]
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
    parser.add_argument("--output-dir", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--output-dim", type=int, help="Output dimension of the embeddings")
    parser.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing (slower, but saves memory)")
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

            # CosineEmbeddingLoss uses the target tensor to determine the objective (pos or neg)
            pos_target = torch.ones(query_emb.size(0), device=query_emb.device)
            neg_target = -torch.ones(query_emb.size(0), device=query_emb.device)
            
            pos_loss = self.loss_fn(query_emb, positive_emb, pos_target)
            neg_loss = self.loss_fn(query_emb, negative_emb, neg_target)
            loss = 2.0 * pos_loss + neg_loss

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

    # Set default output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(DATASET_LOCATION, f"checkpoints-{timestamp}")

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
        save_strategy="no",
        bf16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        label_names=["positive_ids", "positive_mask", "negative_ids", "negative_mask"],
        gradient_checkpointing=args.checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.checkpoint else None,  # More stable checkpointing
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

    class AsyncBestModelCallback(TrainerCallback):
        def __init__(self):
            self.best_model = None
            self.best_score = float('inf')
            self.save_thread = None
            
        def on_evaluate(self, _args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            metrics = kwargs.get("metrics", {})
            eval_loss = metrics.get("eval_loss")
            if eval_loss < self.best_score:
                self.best_score = eval_loss
                self.no_improve_count = 0
                # Store deep copy of model state in memory
                if "model" in kwargs:
                    self.best_model = copy.deepcopy(kwargs["model"].state_dict())
                    
                    # Start background thread to save if previous save is done
                    if self.save_thread is None or not self.save_thread.is_alive():
                        self.save_thread = threading.Thread(
                            target=self._save_model,
                            args=(self.best_model, args.output_dir, state.global_step)
                        )
                        self.save_thread.start()
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= args.patience:
                    print(f"\nNo improvement for {args.patience} epochs. Stopping training.")
                    control.should_training_stop = True
        
        @staticmethod
        def _save_model(model_state, output_dir, step):
            save_path = f"{output_dir}/checkpoint-{step}"
            os.makedirs(save_path, exist_ok=True)
            
            # Split and save base model and projection states
            base_state = {k: v for k, v in model_state.items() if k.startswith('base_model.')}
            projection_state = {k: v for k, v in model_state.items() if k.startswith('projection.')}
            
            torch.save(model_state, f"{save_path}/pytorch_model.bin")
            print(f"\nSaved best model from step {step}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=TripletCollator(tokenizer),
        callbacks=[
            AsyncBestModelCallback()
        ]
    )

    # Train the model
    trainer.train()
    
    # Copy the best checkpoint to final location
    model_name = args.model.split('/')[-1]
    output_path = os.path.join(DATASET_LOCATION, f'fine_tuned_{model_name}_{args.output_dim}_{args.train_files}')
    os.makedirs(output_path, exist_ok=True)
    
    # Find the best checkpoint
    best_checkpoint = None
    best_step = None
    for dirname in os.listdir(args.output_dir):
        if dirname.startswith('checkpoint-'):
            checkpoint_path = os.path.join(args.output_dir, dirname, 'pytorch_model.bin')
            if os.path.exists(checkpoint_path):
                step = int(dirname.split('-')[1])
                if best_step is None or step > best_step:
                    best_checkpoint = checkpoint_path
                    best_step = step

    if best_checkpoint:
        # Move the best checkpoint file
        os.makedirs(output_path, exist_ok=True)
        os.rename(best_checkpoint, os.path.join(output_path, 'pytorch_model.bin'))
        
        # Save the tokenizer
        tokenizer.save_pretrained(output_path)
        print(f"Best model from step {best_step} moved to {output_path}")
        
        # Clean up other checkpoints
        for dirname in os.listdir(args.output_dir):
            if dirname.startswith('checkpoint-'):
                checkpoint_dir = os.path.join(args.output_dir, dirname)
                shutil.rmtree(checkpoint_dir)
        print("Cleaned up unused checkpoints")
    else:
        print("Warning: No checkpoints found to save!")

if __name__ == "__main__":
    main()

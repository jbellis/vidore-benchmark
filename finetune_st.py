import argparse
from datetime import datetime
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import EarlyStoppingCallback

DATASET_LOCATION = "/home/jonathan/datasets/arxivqa"
SEQUENCE_LENGTH = 512


def load_arxiv_dataset(preprocessed_file: str, ocr_dir: str, start_file: int, end_file: int) -> Dataset:
    # Load question mapping
    hash_to_question = {}
    with open(preprocessed_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            hash_to_question[data['image_hash']] = data['question']

    # Load OCR texts
    all_files = sorted(os.listdir(ocr_dir))
    selected_files = all_files[start_file:end_file]
    
    anchors = []  # questions
    positives = []  # matching OCR texts
    all_texts = []  # store all texts for negative sampling
    
    # First pass to collect all texts
    for filename in selected_files:
        with open(os.path.join(ocr_dir, filename), 'r') as f:
            ocr_text = f.read().strip()
            all_texts.append(ocr_text)
            
        file_hash = os.path.splitext(filename)[0]
        if file_hash in hash_to_question:
            question = hash_to_question[file_hash]
            anchors.append(question)
            positives.append(ocr_text)
    
    # Create dataset dictionary
    dataset_dict = {
        'anchor': anchors,
        'positive': positives,
    }
    
    return Dataset.from_dict(dataset_dict)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentence transformer model on ArxivQA dataset")
    parser.add_argument("--train-files", type=int, default=1000, help="Number of files to use for training")
    parser.add_argument("--val-files", type=int, default=100, help="Number of files to use for validation")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-large-en-v1.5", help="Model to fine-tune")
    parser.add_argument("--output-dir", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing (slower, but saves memory)")
    args = parser.parse_args()

    # Calculate gradient accumulation steps: 128/batch_size, clamped between 1 and 64
    gradient_accumulation_steps = min(max(128 // args.batch_size, 1), 64)
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    learning_rate = 2e-5 * effective_batch_size / 64
    print(f"Using LR {learning_rate} with {gradient_accumulation_steps} gradient accumulation steps")

    # Set default output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(DATASET_LOCATION, f"st-checkpoints-{timestamp}")

    # Load datasets
    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')
    
    train_dataset = load_arxiv_dataset(preprocessed_file, ocr_dir, 0, args.train_files)
    val_dataset = load_arxiv_dataset(preprocessed_file, ocr_dir, args.train_files,
                                   args.train_files + args.val_files)

    # Initialize model
    model = SentenceTransformer(args.model,
                                trust_remote_code=True,
                                model_kwargs={"attn_implementation": "flash_attention_2",
                                              "torch_dtype": torch.bfloat16})
    
    # Define loss
    loss = losses.MultipleNegativesRankingLoss(model)

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        bf16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.checkpoint,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    # Initialize trainer with early stopping
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # Train the model
    trainer.train()

    # Save the final model
    model_name = args.model.split('/')[-1]
    output_path = os.path.join(DATASET_LOCATION, f'fine_tuned_{model_name}_{args.train_files}')
    model.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()

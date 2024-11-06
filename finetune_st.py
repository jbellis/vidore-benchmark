import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import EarlyStoppingCallback


def get_dataset_location(dataset: str) -> str:
    dataset_paths = {
        'arxiv': "/home/jonathan/datasets/arxivqa",
        'infovqa': "/home/jonathan/datasets/infovqa",
    }
    if dataset not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset}. Available datasets: {list(dataset_paths.keys())}")
    return dataset_paths[dataset]


def load_infovqa_dataset(annotations_file: str, ocr_dir: str, start_idx: int, end_idx: int) -> Dataset:
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)['data'][start_idx:end_idx]
    
    anchors = []  # questions
    positives = []  # matching OCR texts
    
    for item in data:
        # Get image ID from local name (e.g., "20471.jpeg" -> "20471")
        image_id = os.path.splitext(item['image_local_name'])[0]
        
        # Read OCR text
        ocr_path = os.path.join(ocr_dir, f"{image_id}.txt")
        with open(ocr_path, 'r') as f:
            ocr_text = f.read().strip()
            anchors.append(item['question'])
            positives.append(ocr_text)
    
    return Dataset.from_dict({
        'anchor': anchors,
        'positive': positives,
    })

def load_arxiv_dataset(preprocessed_file: str, ocr_dir: str, start_idx: int, end_idx: int) -> Dataset:
    # Load test questions from HF dataset
    test_dataset = load_dataset("vidore/arxivqa_test_subsampled", split="test")
    test_questions = set(test_dataset["query"])
    
    # Collect questions, grouped by hash
    hash_to_questions = defaultdict(list)
    with open(preprocessed_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            hash_to_questions[data['image_hash']].append(data['question'])

    # Find the hashes of images questions that match test set questions
    test_hashes = set(file_hash
                      for file_hash, questions in hash_to_questions.items()
                      if any(question in test_questions for question in questions))

    from itertools import chain
    # Build the list of questions that do not correspond to test hashes
    train_questions = list(chain.from_iterable(
        ((file_hash, question) for question in questions)
        for file_hash, questions in hash_to_questions.items()
        if file_hash not in test_hashes
    ))
    print(f"{len(train_questions)} questions that are not related to test set")

    # Select questions based on indices
    selected_questions = train_questions[start_idx:end_idx]

    anchors = []  # questions
    positives = []  # matching OCR texts

    # Process selected questions and load corresponding OCR texts
    ocr_cache = {}  # Cache OCR texts to avoid reading the same file multiple times
    for file_hash, question in selected_questions:
        # Load OCR text (with caching)
        if file_hash not in ocr_cache:
            ocr_path = os.path.join(ocr_dir, f"{file_hash}.txt")
            with open(ocr_path, 'r') as f:
                ocr_cache[file_hash] = f.read().strip()
        
        anchors.append(question)
        positives.append(ocr_cache[file_hash])
    
    # Create dataset dictionary
    dataset_dict = {
        'anchor': anchors,
        'positive': positives,
    }
    
    return Dataset.from_dict(dataset_dict)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentence transformer model on ArxivQA dataset",
                                     allow_abbrev=False)  # Disallow abbreviated arguments
    parser.add_argument("--train-files", type=int, default=1000, help="Number of files to use for training")
    parser.add_argument("--val-files", type=int, help="Number of files to use for validation and early stopping")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-large-en-v1.5", help="Model to fine-tune")
    parser.add_argument("--revision", type=str, help="Specific revision of the model to use (e.g., commit hash or branch name)")
    parser.add_argument("--output-dir", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=2, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing (slower, but saves memory)")
    parser.add_argument("--dataset", type=str, default="arxiv", help="Dataset to use for fine-tuning")
    parser.add_argument("--print-data", type=int, help="Print N samples from the dataset")
    parser.add_argument("--eval-steps", type=int, help="Evaluate every N steps. If not set, evaluates every epoch")
    args = parser.parse_args()

    # Calculate gradient accumulation steps: 128/batch_size, clamped between 1 and 64
    gradient_accumulation_steps = min(max(128 // args.batch_size, 1), 64)
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    learning_rate = 2e-5 * effective_batch_size / 64
    print(f"Using LR {learning_rate} with {gradient_accumulation_steps} gradient accumulation steps")

    # Get dataset location
    dataset_location = get_dataset_location(args.dataset)

    # Set default output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(dataset_location, f"st-checkpoints-{timestamp}")

    # Load datasets
    if args.dataset == 'arxiv':
        preprocessed_file = os.path.join(dataset_location, 'preprocessed.jsonl')
        ocr_dir = os.path.join(dataset_location, 'ocr')
        train_dataset = load_arxiv_dataset(preprocessed_file, ocr_dir, 0, args.train_files)
        val_dataset = None
        if args.val_files:
            val_dataset = load_arxiv_dataset(preprocessed_file, ocr_dir, args.train_files,
                                           args.train_files + args.val_files)
    elif args.dataset == 'infovqa':
        annotations_file = os.path.join(dataset_location, 'json/infographicsVQA_train_v1.0.json')
        ocr_dir = os.path.join(dataset_location, 'ocr')
        train_dataset = load_infovqa_dataset(annotations_file, ocr_dir, 0, args.train_files)
        val_dataset = None
        if args.val_files:
            val_dataset = load_infovqa_dataset(annotations_file, ocr_dir, args.train_files, args.val_files)
    print(f"Training dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("No validation dataset")

    # Print dataset samples if requested
    if args.print_data:
        print(f"\nPrinting first {args.print_data} samples from training dataset:")
        for i, sample in enumerate(train_dataset):
            if i >= args.print_data:
                break
            print(f"\nSample {i+1}:")
            print(f"Query: {sample['anchor']}")
            print(f"Document: {sample['positive'][:500]}...")
        return

    # Initialize model. First try with Flash Attention 2.0, then without if that fails
    model_kwargs = {
        "trust_remote_code": True,
        "model_kwargs": {"torch_dtype": torch.bfloat16}
    }
    if args.revision:
        model_kwargs["revision"] = args.revision

    try:
        model_kwargs["model_kwargs"]["attn_implementation"] = "flash_attention_2"
        model = SentenceTransformer(args.model, **model_kwargs)
    except ValueError as e:
        if "Flash Attention" in str(e):
            print("Flash Attention 2.0 not supported, falling back to default attention")
            del model_kwargs["model_kwargs"]["attn_implementation"]
            model = SentenceTransformer(args.model, **model_kwargs)
        else:
            raise e
    
    # Define loss
    loss = losses.MultipleNegativesRankingLoss(model)

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        bf16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.checkpoint,
        eval_strategy="no" if val_dataset is None else "steps" if args.eval_steps else "epoch",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.eval_steps else "epoch",
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=val_dataset is not None,
    )

    # Initialize trainer with early stopping only if validation is enabled
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "loss": loss,
    }
    if val_dataset is not None:
        trainer_kwargs.update({
            "eval_dataset": val_dataset,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.patience)]
        })
    
    trainer = SentenceTransformerTrainer(**trainer_kwargs)

    # Train the model
    trainer.train()

    # Save the final model
    model_name = args.model.split('/')[-1]
    output_path = os.path.join(dataset_location, f'fine_tuned_{model_name}_{args.train_files}')
    model.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()

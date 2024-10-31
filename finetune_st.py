import argparse
import copy
from datetime import datetime
import json
import os
import threading

from datasets import Dataset
from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

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
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    args = parser.parse_args()

    # Set default checkpoint directory if not specified
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.checkpoint_dir = os.path.join(DATASET_LOCATION, f"st-checkpoints-{timestamp}")

    # Check for existing model at target output path
    model_name = args.model.split('/')[-1]
    output_path = os.path.join(DATASET_LOCATION, f'st_fine_tuned_{model_name}_{args.train_files}')
    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}. Skipping training.")
        return

    # Load datasets
    preprocessed_file = os.path.join(DATASET_LOCATION, 'preprocessed.jsonl')
    ocr_dir = os.path.join(DATASET_LOCATION, 'ocr')
    
    train_dataset = load_arxiv_dataset(preprocessed_file, ocr_dir, 0, args.train_files)
    val_dataset = load_arxiv_dataset(preprocessed_file, ocr_dir, args.train_files,
                                   args.train_files + args.val_files)

    # Initialize model
    model = SentenceTransformer(args.model, trust_remote_code=True)
    
    # Define loss
    loss = losses.MultipleNegativesRankingLoss(model)

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="no",
        metric_for_best_model="eval_loss"
    )

    class AsyncBestModelCallback(TrainerCallback):
        def __init__(self):
            self.best_model = None
            self.best_score = float('inf')
            self.save_thread = None
            self.no_improve_count = 0
            
        def on_evaluate(self, _args: SentenceTransformerTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            metrics = kwargs.get("metrics", {})
            eval_loss = metrics.get("eval_loss")
            if eval_loss < self.best_score:
                self.best_score = eval_loss
                self.no_improve_count = 0
                
                if "model" in kwargs:
                    # Create a new SentenceTransformer instance with same config
                    model = kwargs["model"]
                    save_model = SentenceTransformer(model.get_config_dict())
                    # Deep copy the model parameters
                    save_model.load_state_dict({
                        name: param.clone().detach()
                        for name, param in model.state_dict().items()
                    })
                    
                    # Start background thread to save if previous save is done
                    if self.save_thread is None or not self.save_thread.is_alive():
                        self.save_thread = threading.Thread(
                            target=self._save_model,
                            args=(save_model, args.checkpoint_dir, state.global_step)
                        )
                        self.save_thread.start()
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= args.patience:
                    print(f"\nNo improvement for {args.patience} epochs. Stopping training.")
                    control.should_training_stop = True
        
        @staticmethod
        def _save_model(model, output_dir, step):
            save_path = f"{output_dir}/checkpoint-{step}"
            os.makedirs(save_path, exist_ok=True)
            model.save(save_path)
            print(f"\nSaved best model checkpoint from step {step}")
            
        def cleanup_checkpoints(self, output_dir):
            """Remove all checkpoint directories"""
            import shutil
            for dirname in os.listdir(output_dir):
                if dirname.startswith('checkpoint-'):
                    checkpoint_dir = os.path.join(output_dir, dirname)
                    shutil.rmtree(checkpoint_dir)
            print("Cleaned up unused checkpoints")

        def promote_or_save_best_checkpoint(self, model, output_path):
            """Promote the best checkpoint to final output location"""
            # Find the best checkpoint
            best_checkpoint = None
            best_step = None
            for dirname in os.listdir(args.checkpoint_dir):
                if dirname.startswith('checkpoint-'):
                    checkpoint_path = os.path.join(args.checkpoint_dir, dirname)
                    if os.path.exists(checkpoint_path):
                        step = int(dirname.split('-')[1])
                        if best_step is None or step > best_step:
                            best_checkpoint = checkpoint_path
                            best_step = step

            if best_checkpoint:
                # Load step number
                with open(os.path.join(best_checkpoint, 'step.txt'), 'r') as f:
                    step = int(f.read().strip())
                
                # Copy the complete model to final location
                import shutil
                shutil.copytree(best_checkpoint, output_path, dirs_exist_ok=True)
                print(f"Promoted best checkpoint from step {step} to {output_path}")
            else:
                # Save the current model state if no checkpoints found
                os.makedirs(output_path, exist_ok=True)
                model.save(output_path)
                print(f"Saved final model state to {output_path}")


    # Create callback
    callback = AsyncBestModelCallback()

    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        callbacks=[callback]
    )

    # Train the model
    trainer.train()

    callback.promote_or_save_best_checkpoint(model, output_path)
    callback.cleanup_checkpoints(args.checkpoint_dir)

if __name__ == "__main__":
    main()

import os
import random
import time
from typing import List, Tuple

import torch
from tqdm import tqdm

from finetune_st import load_arxiv_dataset
from modeling_voyage_q import VoyageQForSequenceClassification
from transformers import AutoTokenizer


class VoyageLocalReranker:
    def __init__(self, device="cuda"):
        model_path = "/mnt/T9/models/voyage-rerank-2-lite/rerank-2-lite"
        self.max_length = 32_000
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=self.max_length, use_fast=True)
        self.model = VoyageQForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device)
        self.model.eval()

    def rerank(self, query: str, documents_to_rerank: list[str]) -> dict[int, float]:
        pairs = [f"query: {query} \n \n passage: {doc}" for doc in documents_to_rerank]

        encoded_input = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            verbose=False,
        )
        
        # Truncate to max_length
        # TODO: why is this slower than just calling model(**encoded_input)?
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)
        input_ids = input_ids[:, :self.max_length]
        attention_mask = attention_mask[:, :self.max_length]
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits).tolist()

        # Create dictionary mapping indices 0..N-1 to scores
        return {i: score for i, score in enumerate(scores)}


def rerank_random(reranker, dataset):
    """Rerank 1 random query against 40 random passages"""
    
    # Select random query and passages
    query = random.choice(dataset['anchor'])
    passages = random.sample(dataset['positive'], 40)
    
    # Get reranking scores
    return reranker.rerank(query, passages)

def main():
    """Time 100 random reranking operations"""
    # Load model
    model_name = "rerank-2"
    root_path = "./"
    dataset_path = "/home/jonathan/datasets/arxivqa"
    reranker = VoyageLocalReranker()
    
    # Load dataset once
    preprocessed_file = os.path.join(dataset_path, 'preprocessed.jsonl')
    ocr_dir = os.path.join(dataset_path, 'ocr')
    dataset = load_arxiv_dataset(preprocessed_file, ocr_dir, 0, 100000)
    
    times = []
    for _ in tqdm(range(20)):
        start = time.time()
        rerank_random(reranker, dataset)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average time per rerank: {avg_time:.3f} seconds")

if __name__ == "__main__":
    main()

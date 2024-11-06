import os
import time
from abc import abstractmethod, ABC
from pathlib import Path

import torch
import voyageai


class RerankProvider(ABC):
    @abstractmethod
    def rerank(self,
               query: str,
               documents_to_rerank: list[str],
               document_ids: list[int]
               ) -> dict[int, float]:
        pass


class CohereRerankProvider(RerankProvider):
    def __init__(self):
        import cohere
        self.cohere_client = cohere.Client(api_key=os.environ.get('COHERE_API_KEY'))

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int]) -> dict[int, float]:
        reranked_results = self.cohere_client.rerank(
            query=query,
            documents=documents_to_rerank,
            model='rerank-multilingual-v3.0',
        )

        return {document_ids[int(result.index)]: result.relevance_score
                for result in reranked_results.results}


class JinaRerankProvider(RerankProvider):
    def __init__(self, device):
        from transformers import AutoModelForSequenceClassification
        jina_model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            torch_dtype="auto",
            trust_remote_code=True,
        )
        jina_model.to(device)
        jina_model.eval()
        self.jina_model = jina_model

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int]) -> dict[int, float]:
        with torch.no_grad():
            sentence_pairs = [[query, doc] for doc in documents_to_rerank]
            scores = self.jina_model.compute_score(sentence_pairs, max_length=1024)

        return {doc_id: scores[idx] for idx, doc_id in enumerate(document_ids)}


class VoyageRerankProvider(RerankProvider):
    def __init__(self):
        import voyageai
        self.voyage_client = voyageai.Client(api_key=os.environ.get('VOYAGE_API_KEY'))

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int]) -> dict[int, float]:
        # Filter out empty documents and keep track of original indices
        filtered_documents = []
        filtered_indices = []
        for idx, doc in enumerate(documents_to_rerank):
            if doc.strip():
                filtered_documents.append(doc)
                filtered_indices.append(idx)

        backoff = 1.0
        while True:
            try:
                reranked_results = self.voyage_client.rerank(
                    query=query,
                    documents=filtered_documents,
                    model="rerank-2",
                    truncation=False
                )
            except voyageai.error.RateLimitError:
                print(f'Rate limit error. Waiting {backoff} seconds and trying again.')
                time.sleep(backoff)
                backoff *= 2
            else:
                break

        reranked_scores = {}
        for result in reranked_results.results:
            original_index = filtered_indices[result.index]
            doc_id = document_ids[original_index]
            reranked_scores[doc_id] = result.relevance_score

        return reranked_scores


class BGERerankProvider(RerankProvider):
    def __init__(self):
        from FlagEmbedding import FlagReranker
        self.bge_reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int]) -> dict[int, float]:
        # Prepare input for BGE reranker
        rerank_input = [[query, doc] for doc in documents_to_rerank]

        # Compute scores
        scores = self.bge_reranker.compute_score(rerank_input, normalize=True)

        # Create the final reranked_scores dictionary using dict comprehension
        return {doc_id: scores[idx] for idx, doc_id in enumerate(document_ids)}


class RRFRerankProvider(RerankProvider):
    def __init__(self, k: int = 20):
        self.k = k

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int],
               list_emb_documents: list[str]) -> dict[str, float]:
        # This method needs to be adapted to work with the new structure
        # It currently relies on bm25_top_20 and dpr_scores_indexed which are not passed as parameters
        # For now, we'll leave it as a placeholder
        raise NotImplementedError("RRF reranking needs to be adapted to the new structure")


class NvidiaRerankProvider(RerankProvider):
    def __init__(self, device="cuda"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path = "/home/jonathan/Projects/nvidia/nv-rerank-qa_vllama-3.2-nv-rerankqa-1B-v1"
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int]) -> dict[int, float]:
        pairs = [(query, doc) for doc in documents_to_rerank]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            results = self.model(**inputs)
            logits = results.logits.squeeze(-1)
            scores = torch.sigmoid(logits).tolist()

        return {doc_id: score for doc_id, score in zip(document_ids, scores)}

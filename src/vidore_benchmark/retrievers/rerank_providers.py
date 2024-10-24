import time
from abc import abstractmethod, ABC

import torch
import voyageai


class RerankProvider(ABC):
    @abstractmethod
    def rerank(self, query: str, documents_to_rerank: list[str], combined_ordinals: list[int],
               list_emb_documents: list[str]) -> dict[str, float]:
        pass


class CohereRerankProvider(RerankProvider):
    def __init__(self, cohere_client):
        self.cohere_client = cohere_client

    def rerank(self, query: str, documents_to_rerank: list[str], combined_ordinals: list[int],
               list_emb_documents: list[str]) -> dict[str, float]:
        reranked_results = self.cohere_client.rerank(
            query=query,
            documents=documents_to_rerank,
            model='rerank-multilingual-v3.0',
            top_n=5
        )

        reranked_scores = {doc_id: 0.0 for doc_id in list_emb_documents}
        for result in reranked_results.results:
            doc_id = combined_ordinals[int(result.index)]
            reranked_scores[list_emb_documents[doc_id]] = result.relevance_score

        return reranked_scores


class JinaRerankProvider(RerankProvider):
    def __init__(self, jina_model):
        self.jina_model = jina_model

    def rerank(self, query: str, documents_to_rerank: list[str], combined_ordinals: list[int],
               list_emb_documents: list[str]) -> dict[str, float]:
        with torch.no_grad():
            sentence_pairs = [[query, doc] for doc in documents_to_rerank]
            scores = self.jina_model.compute_score(sentence_pairs, max_length=1024)

        reranked_scores = {doc_id: 0.0 for doc_id in list_emb_documents}
        for idx, doc_id in enumerate(combined_ordinals):
            reranked_scores[list_emb_documents[doc_id]] = scores[idx]

        return reranked_scores


class VoyageRerankProvider(RerankProvider):
    def __init__(self, voyage_client):
        self.voyage_client = voyage_client

    def rerank(self, query: str, documents_to_rerank: list[str], combined_ordinals: list[int],
               list_emb_documents: list[str]) -> dict[str, float]:
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

        reranked_scores = {doc_id: 0.0 for doc_id in list_emb_documents}
        for result in reranked_results.results:
            original_index = filtered_indices[result.index]
            doc_id = combined_ordinals[original_index]
            reranked_scores[list_emb_documents[doc_id]] = result.relevance_score

        return reranked_scores


class BGERerankProvider(RerankProvider):
    def __init__(self, bge_reranker):
        self.bge_reranker = bge_reranker

    def rerank(self, query: str, documents_to_rerank: list[str], combined_ordinals: list[int],
               list_emb_documents: list[str]) -> dict[str, float]:
        # Prepare input for BGE reranker
        rerank_input = [[query, doc] for doc in documents_to_rerank]

        # Compute scores
        scores = self.bge_reranker.compute_score(rerank_input, normalize=True)

        # Create the final reranked_scores dictionary
        reranked_scores = {doc_id: 0.0 for doc_id in list_emb_documents}
        for idx, doc_id in enumerate(combined_ordinals):
            reranked_scores[list_emb_documents[doc_id]] = scores[idx]

        return reranked_scores


class RRFRerankProvider(RerankProvider):
    def __init__(self, k: int = 20):
        self.k = k

    def rerank(self, query: str, documents_to_rerank: list[str], combined_ordinals: list[int],
               list_emb_documents: list[str]) -> dict[str, float]:
        # This method needs to be adapted to work with the new structure
        # It currently relies on bm25_top_20 and dpr_scores_indexed which are not passed as parameters
        # For now, we'll leave it as a placeholder
        raise NotImplementedError("RRF reranking needs to be adapted to the new structure")

import os
import time
from abc import abstractmethod, ABC
from pathlib import Path
import torch
import voyageai
from openai import OpenAI


def deepseek_client() -> OpenAI:
    """Create and return an OpenAI client configured for Deepseek."""
    deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
    return OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")


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


class LlmRerankProvider(RerankProvider):
    def __init__(self, client: OpenAI):
        self.client = client

    def _rerank_window(self, query: str, documents: list[str], document_indices: list[int]) -> list[int]:
        """Rerank a single window of documents using DeepSeek API and return ordered indices."""
        # Create prompt with numbered passages
        passages_text = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(documents))
        
        messages = [
            {"role": "system", "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {"role": "user", "content": f"I will provide you with {len(documents)} passages. Rank them based on their relevance to query: {query}"},
            {"role": "assistant", "content": "Okay, please provide the passages."},
            {"role": "user", "content": passages_text},
            {"role": "user", "content": f"Search Query: {query}. Rank the passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first. Only respond with the ranking results, do not say any other words or explain."}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        
        # Parse ranking from response
        ranking_text = response.choices[0].message.content.strip()
        # Extract just the numbers from the response, ignoring all separators
        import re
        ranks = [int(x) for x in re.findall(r'\d+', ranking_text)]
        if len(ranks) != len(documents):
            raise Exception(f"Response {response} did not include all {len(documents)} candidates")
            
        # Convert ranks to ordered indices
        ordered_indices = [document_indices[rank-1] for rank in ranks]
        print(ordered_indices)
        return ordered_indices

class SlidingWindowRerankProvider(LlmRerankProvider):
    def __init__(self, client: OpenAI, window_size: int = 20, step_size: int = 10):
        """
        Initialize LLM reranker with sliding window parameters.

        Args:
            client: OpenAI client instance
            window_size: Number of passages to rerank in each window
            step_size: Number of passages to slide the window by. Windows will overlap
                      by (window_size - step_size) passages to allow averaging scores
                      across multiple windows.
        """
        super().__init__(client)
        self.window_size = window_size
        self.step_size = step_size

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int]) -> dict[int, float]:
        """
        Rerank documents using sliding window approach.
        
        Each window consists of:
        1. A new slice of step_size documents
        2. The top (window_size - step_size) documents from the previous window
        
        Returns a dictionary mapping document IDs to scores based on their final ranking position.
        """
        num_docs = len(documents_to_rerank)
        if num_docs <= self.window_size:
            # If fewer documents than window size, rerank all at once
            ordered_indices = self._rerank_window(query, documents_to_rerank, document_ids)
            return {doc_id: len(document_ids) - i for i, doc_id in enumerate(ordered_indices)}
            
        # Initialize with last window
        start_idx = num_docs - self.window_size
        window_docs = documents_to_rerank[start_idx:]
        window_ids = document_ids[start_idx:]
        top_indices = self._rerank_window(query, window_docs, window_ids)
        
        # Keep track of top docs from previous window
        carry_size = self.window_size - self.step_size
        
        # Process remaining documents in sliding windows from back to front
        for start_idx in range(num_docs - self.window_size - self.step_size, -1, -self.step_size):
            # Get new documents for this window
            end_idx = start_idx + self.step_size
            new_docs = documents_to_rerank[start_idx:end_idx]
            new_ids = document_ids[start_idx:end_idx]
            
            # Get top docs from previous window
            top_docs = [documents_to_rerank[i] for i in range(len(documents_to_rerank)) if document_ids[i] in top_indices[:carry_size]]
            top_ids = top_indices[:carry_size]
            
            # Combine new docs with top docs from previous window
            window_docs = new_docs + top_docs
            window_ids = new_ids + top_ids
            
            # Rerank current window
            top_indices = self._rerank_window(query, window_docs, window_ids)
        
        # Convert final ordering to scores
        final_scores = {doc_id: len(document_ids) - i for i, doc_id in enumerate(top_indices)}
        return final_scores

class DeepseekSlidingWindowRerankProvider(SlidingWindowRerankProvider):
    def __init__(self, window_size: int = 20, step_size: int = 10):
        super().__init__(deepseek_client(), window_size, step_size)


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

        # Enable DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def rerank(self, query: str, documents_to_rerank: list[str], document_ids: list[int]) -> dict[int, float]:
        pairs = [f"query: {query} \n \n passage: {doc}" for doc in documents_to_rerank]

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

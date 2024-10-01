#
from typing import List, Optional, Any, Tuple, Dict
import torch
from PIL import Image

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import Model
from colbert_live.db.astra import AstraCQL

class ColbertLiveDB(AstraCQL):
    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token, verbose=True)

    def prepare(self, embedding_dim: int):
        # Create tables and indexes
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.documents (
                doc_id uuid PRIMARY KEY,
                content blob
            )
        """)
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.embeddings (
                doc_id uuid,
                embedding_id int,
                embedding vector<float, {embedding_dim}>,
                PRIMARY KEY ((doc_id), embedding_id)
            )
        """)
        self.session.execute(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann 
            ON {self.keyspace}.embeddings(embedding) 
            USING 'StorageAttachedIndex'
        """)

        # Prepare statements
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT doc_id, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.embeddings
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.embeddings WHERE doc_id = ?
        """)

    def process_ann_rows(self, result):
        return [(row.doc_id, row.similarity) for row in result]

    def process_chunk_rows(self, result):
        return [torch.tensor(row.embedding) for row in result]

@register_vision_retriever("colbert_live")
class ColbertLiveRetriever(VisionRetriever):
    """
    ColbertLiveRetriever class to retrieve embeddings using ColbertLive.
    """

    def __init__(self, device: str = "auto", model_name: str = "vidore/colpali-v1.2", 
                 astra_db_id: str = None, astra_token: str = None):
        super().__init__()
        self.device = get_torch_device(device)
        self.model = Model.from_name_or_path(model_name)
        self.db = ColbertLiveDB('colbert_live', self.model.dim, astra_db_id, astra_token)
        self.colbert_live = ColbertLive(self.db, self.model)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        return self.colbert_live.encode_query(queries)

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        return self.colbert_live.encode_chunks(documents)

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        scores = []
        for query_emb in list_emb_queries:
            query_scores = self.colbert_live.search(query_emb, k=len(list_emb_documents), n_ann_docs=batch_size)
            scores.append([score for _, score in query_scores])
        return torch.tensor(scores)

    def get_relevant_docs_results(
        self,
        ds: Any,
        queries: List[str],
        scores: torch.Tensor,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        relevant_docs = {}
        results = {}

        for query, score_per_query in zip(queries, scores):
            relevant_docs[query] = {str(ds["image_filename"][0]): 1}  # Assuming first document is relevant

            for doc_idx, score in enumerate(score_per_query):
                filename = str(ds["image_filename"][doc_idx])
                score_passage = float(score.item())

                if query in results:
                    results[query][filename] = max(results[query].get(filename, 0), score_passage)
                else:
                    results[query] = {filename: score_passage}

        return relevant_docs, results

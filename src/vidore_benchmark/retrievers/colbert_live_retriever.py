from typing import List, Optional, Any, Tuple, Dict
import torch
from PIL import Image
import os
import logging
import psutil
from tqdm import tqdm

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import Model
from colbert_live.db.astra import AstraCQL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        keyspace = os.environ.get('VIDORE_KEYSPACE')
        if not keyspace:
            raise ValueError("VIDORE_KEYSPACE environment variable is not set")
        
        self.db = ColbertLiveDB(keyspace, self.model.dim, astra_db_id, astra_token)
        self.colbert_live = ColbertLive(self.db, self.model)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        logger.info(f"Encoding {len(queries)} queries")
        return self.colbert_live.encode_query(queries)

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        logger.info(f"Encoding {len(documents)} documents")
        all_embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
            batch = documents[i:i+batch_size]
            try:
                batch_embeddings = self.colbert_live.encode_chunks(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size}: {str(e)}")
            
            # Log memory usage
            process = psutil.Process(os.getpid())
            logger.info(f"Memory usage after batch {i//batch_size}: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        return all_embeddings

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        logger.info(f"Computing scores for {len(list_emb_queries)} queries and {len(list_emb_documents)} documents")
        scores = []
        for query_emb in tqdm(list_emb_queries, desc="Computing scores"):
            try:
                query_scores = self.colbert_live.search(query_emb, k=len(list_emb_documents), n_ann_docs=batch_size)
                scores.append([score for _, score in query_scores])
            except Exception as e:
                logger.error(f"Error computing scores for query: {str(e)}")
                scores.append([0.0] * len(list_emb_documents))  # Add dummy scores in case of error
        return torch.tensor(scores)

    def get_relevant_docs_results(
        self,
        ds: Any,
        queries: List[str],
        scores: torch.Tensor,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        logger.info("Processing relevant docs and results")
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

from typing import List, Optional, Any, Tuple, Dict
import torch
from PIL import Image
import os
import logging
import psutil
from tqdm import tqdm
import io
import uuid
from cassandra.concurrent import execute_concurrent_with_args

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
        self.insert_document_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.documents (doc_id, content) VALUES (?, ?)
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.embeddings (doc_id, embedding_id, embedding) VALUES (?, ?, ?)
        """)

    def process_ann_rows(self, result):
        return [(row.doc_id, row.similarity) for row in result]

    def process_chunk_rows(self, result):
        return [torch.tensor(row.embedding) for row in result]

    def add_documents(self, contents: List[bytes], embeddings_list: List[torch.Tensor]) -> List[uuid.UUID]:
        doc_ids = [uuid.uuid4() for _ in contents]
        
        # Insert documents
        document_params = list(zip(doc_ids, contents))
        execute_concurrent_with_args(self.session, self.insert_document_stmt, document_params)

        # Insert embeddings
        embedding_params = []
        for doc_id, doc_embeddings in zip(doc_ids, embeddings_list):
            for embedding_id, embedding in enumerate(doc_embeddings):
                embedding_params.append((doc_id, embedding_id, embedding.tolist()))
        
        execute_concurrent_with_args(self.session, self.insert_embedding_stmt, embedding_params)

        return doc_ids

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
        logger.info(f"Encoding {len(queries)} queries with batch_size={batch_size}")
        encoded_queries = []
        for query in tqdm(queries, desc="Encoding queries"):
            encoded_query = self.colbert_live.encode_query(query)
            encoded_queries.append(encoded_query)
        return encoded_queries

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        logger.info(f"Encoding and storing {len(documents)} documents")
        all_embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents"):
            batch = documents[i:i+batch_size]
            batch_embeddings = self.colbert_live.encode_chunks(batch)
            
            # Convert PIL Images to bytes
            batch_contents = []
            for doc in batch:
                with io.BytesIO() as output:
                    doc.save(output, format="PNG")
                    batch_contents.append(output.getvalue())
            
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        scores = self.colbert_live.model.processor.score(
            list_emb_queries,
            list_emb_documents,
            batch_size=batch_size,
            device=self.device,
        )
        return scores
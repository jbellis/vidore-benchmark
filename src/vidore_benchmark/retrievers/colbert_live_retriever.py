from typing import List, Optional, Any, Tuple, Dict
import torch
from PIL import Image
import os
import logging
import psutil
from tqdm import tqdm
import io
import hashlib
from colbert_live.db.astra import execute_concurrent_async
from itertools import cycle

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
                doc_id text PRIMARY KEY,
                content blob
            )
        """)
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.embeddings (
                doc_id text,
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

    def add_documents(self, contents: List[bytes], embeddings_list: List[torch.Tensor]) -> List[str]:
        doc_ids = [hashlib.sha256(content).hexdigest() for content in contents]
        
        # Prepare statements with parameters
        document_stmt_with_params = list(zip(cycle([self.insert_document_stmt]), zip(doc_ids, contents)))
        
        embedding_stmt_with_params = []
        for doc_id, doc_embeddings in zip(doc_ids, embeddings_list):
            for embedding_id, embedding in enumerate(doc_embeddings):
                embedding_stmt_with_params.append((self.insert_embedding_stmt, (doc_id, embedding_id, embedding.tolist())))
        
        # Execute statements asynchronously
        future = execute_concurrent_async(self.session, document_stmt_with_params + embedding_stmt_with_params)
        
        return doc_ids, future

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
        doc_pool_factor = int(os.environ.get('VIDORE_DOC_POOL', 0))
        query_pool_distance = float(os.environ.get('VIDORE_QUERY_POOL', 0.0))
        
        self.db = ColbertLiveDB(keyspace, self.model.dim, astra_db_id, astra_token)
        self.colbert_live = ColbertLive(self.db, self.model, doc_pool_factor, query_pool_distance)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        logger.info(f"Encoding {len(queries)} queries with batch_size={batch_size}")
        encoded_queries = []
        for query in tqdm(queries, desc="Encoding queries"):
            encoded_query = self.colbert_live.encode_query(query)
            encoded_queries.append(encoded_query)
        print(len(encoded_queries), 'sample query embedding dimensions', encoded_queries[0].shape)
        return encoded_queries

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[str]:
        # encode to bytes
        document_bytes = []
        for doc in documents:
            with io.BytesIO() as output:
                doc.save(output, format="PNG")
                document_bytes.append(output.getvalue())

        result = self.colbert_live.db.session.execute(f"SELECT doc_id FROM {self.colbert_live.db.keyspace}.documents LIMIT 1")
        if result.one() is not None:
            logger.info("Database is not empty, skipping document encoding")
            return [hashlib.sha256(content).hexdigest() for content in document_bytes]

        logger.info(f"Encoding and storing {len(documents)} documents")
        all_doc_ids = []
        future = None
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents"):
            # Encode the current batch
            batch = documents[i:i+batch_size]
            batch_contents = document_bytes[i:i+batch_size]
            batch_embeddings = self.colbert_live.encode_chunks(batch)

            # Wait for the previous batch to complete
            if future:
                future.result()

            # Add documents and their embeddings to the database
            doc_ids, future = self.db.add_documents(batch_contents, batch_embeddings)
            all_doc_ids.extend(doc_ids)

        # Wait for the last batch to complete
        if future:
            future.result()

        print('sample doc embedding dimensions', batch_embeddings[0].shape)
        return all_doc_ids

    def get_scores(
            self,
            list_emb_queries: List[torch.Tensor],
            list_emb_documents: List[str],
            batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        logger.info(f"Computing scores for {len(list_emb_queries)} queries and {len(list_emb_documents)} documents")
        scores = []
        for query_emb in tqdm(list_emb_queries, desc="Computing scores"):
            query_scores = self.colbert_live._search(query_emb, 5, None, None)
            score_dict = dict(query_scores)
            query_scores = [score_dict.get(doc_id, 0.0) for doc_id in list_emb_documents]
            scores.append(query_scores)
        return torch.tensor(scores)

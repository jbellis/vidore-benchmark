from typing import Optional
import torch
from PIL import Image
import os
import logging
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT, colbert_score_reduce
from colbert.search.strided_tensor import StridedTensor
from tqdm import tqdm
import io
import hashlib
from typing import List
from colbert_live.db.astra import execute_concurrent_async
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import Model
from colbert_live.db.astra import AstraCQL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_to_bytes(doc: Image.Image) -> bytes:
    # Calculate new dimensions (half of the original)
    width, height = doc.size
    new_width, new_height = width // 2, height // 2
    
    # Resize the image
    resized_doc = doc.resize((new_width, new_height), Image.LANCZOS)
    
    # Save to bytes
    with io.BytesIO() as output:
        resized_doc.save(output, format="PNG")
        return output.getvalue()

class ColbertLiveDB(AstraCQL):
    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token)

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

class CpuModelStub(Model):
    def __init__(self):
        super().__init__()
        ColBERT.try_load_torch_extensions(False)

    def encode_query(self, q):
        raise NotImplementedError()

    def encode_doc(self, chunks):
        raise NotImplementedError()

    def to_device(self, T):
        return T.cpu()

    @property
    def dim(self):
        return 128

@register_vision_retriever("colbert_live")
class ColbertLiveRetriever(VisionRetriever):
    """
    ColbertLiveRetriever class to retrieve embeddings using ColbertLive.
    """

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)
        self.doc_pool_factor = int(os.environ.get('VIDORE_DOC_POOL'))
        self.n_ann_docs = int(os.environ.get('VIDORE_N_ANN'))
        self.n_maxsim_candidates = int(os.environ.get('VIDORE_N_MAXSIM'))
        self.query_pool_distance = float(os.environ.get('VIDORE_QUERY_POOL'))
        self.cache_dir = os.path.join(os.getcwd(), 'query_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = None

    def use_dataset(self, ds):
        self.dataset_name = ds.name
        keyspace = self.keyspace_name(ds.name)

        if not self.model:
            self.model = CpuModelStub()
        self.db = ColbertLiveDB(keyspace, self.model.dim, None, None)
        self.colbert_live = ColbertLive(self.db, self.model, self.doc_pool_factor, self.query_pool_distance)

    def keyspace_name(self, dataset_name):
        return ''.join([c if c.isalnum() else '_' for c in dataset_name.lower()]) + '_' + str(self.doc_pool_factor)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        logger.info(f"Encoding {len(queries)} queries with batch_size={batch_size}")
        
        encoded_queries = []
        for query in tqdm(queries, desc="Encoding queries"):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"{query_hash}.pt")
            
            if os.path.exists(cache_file):
                encoded_query = self.model.to_device(torch.load(cache_file))
            else:
                encoded_query = self.colbert_live.encode_query(query)
                torch.save(encoded_query, cache_file)
            
            encoded_queries.append(encoded_query)
        
        logger.info(f"Loaded/Encoded {len(encoded_queries)} queries")
        print(len(encoded_queries), 'sample query embedding dimensions', encoded_queries[0].shape)
        return encoded_queries

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[str]:
        with ThreadPoolExecutor() as executor:
            document_bytes = list(tqdm(executor.map(encode_to_bytes, documents), total=len(documents), desc="Encoding to bytes"))

        result = self.colbert_live.db.session.execute(f"SELECT doc_id FROM {self.colbert_live.db.keyspace}.documents LIMIT 1")
        if result.one() is not None:
            logger.info("Database is not empty, skipping document encoding")

            def compute_sha256(content):
                return hashlib.sha256(content).hexdigest()

            with ThreadPoolExecutor() as executor:
                return list(tqdm(executor.map(compute_sha256, document_bytes), total=len(document_bytes), desc="Computing SHA256"))

        logger.info(f"Encoding and storing {len(documents)} documents")
        all_doc_ids = []
        future = None
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents"):
            # Encode the current batch
            batch = documents[i:i+batch_size]
            batch_bytes = document_bytes[i:i+batch_size]
            batch_embeddings = self.colbert_live.encode_chunks(batch)

            # Wait for the previous batch to complete
            if future:
                future.result()

            # Add documents and their embeddings to the database
            doc_ids, future = self.db.add_documents(batch_bytes, batch_embeddings)
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
            query_scores = self.colbert_live._search(query_emb, 100, n_ann_docs=self.n_ann_docs, n_maxsim_candidates=self.n_maxsim_candidates, params={})
            score_dict = dict(query_scores)
            query_scores = [score_dict.get(doc_id, 0.0) for doc_id in list_emb_documents]
            scores.append(query_scores)
        return torch.tensor(scores)

        # # Default scores with the correct shape
        # return torch.ones((len(list_emb_queries), len(list_emb_documents)))

    def get_save_one_path(self, output_path, dataset_name):
        fname = f'{self.keyspace_name(dataset_name)}_{self.query_pool_distance}_{self.n_ann_docs}_{self.n_maxsim_candidates}.pth'
        return os.path.join(output_path, fname)

    def get_save_all_path(self, output_path):
        return self.get_save_one_path(output_path, 'all')

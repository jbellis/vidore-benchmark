import hashlib
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from openai import OpenAI, embeddings
import tiktoken
import torch
from PIL import Image
from cassandra.cluster import Session, Cluster
from colbert_live.db.astra import execute_concurrent_async
from google.api_core.exceptions import InternalServerError
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

import google.generativeai as genai

from .colbert_live_retriever import encode_to_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class DprDB:
    def __init__(self, keyspace: str, vector_size: int):
        self.keyspace = keyspace
        self.vector_size = vector_size
        self.session = self._connect_astra()
        self._create_schema()

    def _connect_astra(self) -> Session:
        cluster = Cluster()
        return cluster.connect()

    def _create_schema(self):
        self.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
        """)
        self.session.execute(f"USE {self.keyspace}")
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id text PRIMARY KEY,
                text text,
                embedding vector<float, %s>
            )
        """, [self.vector_size])
        self.session.execute(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS ON documents (embedding) 
            USING 'StorageAttachedIndex'
        """)
        self.insert_stmt = self.session.prepare("""
            INSERT INTO documents (id, text, embedding)
            VALUES (?, ?, ?)
        """)
        self.search_stmt = self.session.prepare("""
            SELECT id, text, similarity_cosine(embedding, ?) AS similarity
            FROM documents
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)

    def insert_documents(self, doc_params: list[tuple[str, str, list[float]]]):
        futures = execute_concurrent_async(self.session, [(self.insert_stmt, params) for params in doc_params])
        return futures

    def search(self, query_embedding: list[float], k: int) -> list[tuple[str, float]]:
        rows = self.session.execute(self.search_stmt, (query_embedding, query_embedding, k))
        return [(row.id, row.similarity) for row in rows]

    def document_exists(self, doc_id: str) -> bool:
        query = "SELECT 1 FROM documents WHERE id = %s"
        rows = self.session.execute(query, (doc_id,))
        return rows.one() is not None


def get_embeddings(provider, texts: list[str], is_query: bool = False) -> list[list[float]]:
    if provider.startswith('openai'):
        tiktoken_model = tiktoken.encoding_for_model('text-embedding-3-small')
        def tokenize(text: str) -> list[int]:
            return tiktoken_model.encode(text, disallowed_special=())
        def token_length(text: str) -> int:
            return len(list(tokenize(text)))
        def truncate_to(text, max_tokens):
            truncated_tokens = list(tokenize(text))[:max_tokens]
            truncated_s = tiktoken_model.decode(truncated_tokens)
            return truncated_s
        truncated_texts = []
        for text in texts:
            if token_length(text) > 8000:
                global truncated_passages
                truncated_passages += 1
                text = truncate_to(text, 8000)
            truncated_texts.append(text)
        response = openai_client.embeddings.create(
            input=truncated_texts,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]
    elif provider.startswith('gemini'):
        model = "models/text-embedding-004"
        result = genai.embed_content(model=model, content=texts)
        return result['embedding']
    elif provider.startswith('stella'):
        global STELLA_MODEL
        if STELLA_MODEL is None:
            STELLA_MODEL = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()
        if is_query:
            return STELLA_MODEL.encode(texts, prompt_name="s2p_query").tolist()
        else:
            return STELLA_MODEL.encode(texts).tolist()
    else:
        raise ValueError(f"Invalid embedding provider: {provider}")


@register_vision_retriever("dpr_sherpa")
class DprSherpaRetriever(VisionRetriever):
    """
    DprSherpaRetriever class to retrieve embeddings using a DPR embeddings model with llm_sherpa
    chunking the images.
    """

    def __init__(self, device: str = "auto", gemini_api_key: str = None):
        super().__init__()
        self.device = get_torch_device(device)
        self.query_cache_dir = os.path.join(os.getcwd(), 'query_cache')
        os.makedirs(self.query_cache_dir, exist_ok=True)
        self.document_cache_dir = os.path.join(os.getcwd(), 'document_cache')
        os.makedirs(self.document_cache_dir, exist_ok=True)
        self.model_name = 'openai-v3-small'
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')
        self.db = None # initialized by use_dataset

    def use_dataset(self, ds):
        self.db = DprDB(self.keyspace_name(ds.name), 1536)

    def keyspace_name(self, dataset_name):
        return ''.join([c if c.isalnum() else '_' for c in (dataset_name + '_' + self.model_name).lower()])

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: list[str], batch_size: int, **kwargs) -> list[torch.Tensor]:
        logger.info(f"Encoding {len(queries)} queries with batch_size={batch_size}")
        
        encoded_queries = []
        for query in tqdm(queries, desc="Encoding queries"):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            cache_file = os.path.join(self.query_cache_dir, f"{query_hash}.pt")
            if not os.path.exists(cache_file):
                raise NotImplementedError('Unable to recompute query embeddings')

            encoded_query = torch.load(cache_file)
            encoded_queries.append(encoded_query)
        
        logger.info(f"Loaded/Encoded {len(encoded_queries)} queries")
        print(len(encoded_queries), 'sample query embedding dimensions', encoded_queries[0].shape)
        return encoded_queries

    def forward_documents(self, documents: list[Image.Image], batch_size: int, **kwargs) -> list[str]:
        with ThreadPoolExecutor() as executor:
            document_bytes = list(tqdm(executor.map(encode_to_bytes, documents), total=len(documents), desc="Encoding to bytes"))

        def compute_sha256(content):
            return hashlib.sha256(content).hexdigest()
        with ThreadPoolExecutor() as executor:
            document_hashes = list(executor.map(compute_sha256, document_bytes))

        logger.info(f"Chunking {len(documents)} documents")

        encoding_errors = 0
        copyright_errors = 0
        server_errors = 0
        doc_texts = []
        for doc_image, doc_hash in tqdm(zip(documents, document_hashes), total=len(documents), desc="OCR-ing documents"):
            cache_file = os.path.join(self.document_cache_dir, f"{doc_hash}.txt")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    extracted_text = json.load(f)
                logger.info(f"Loaded cached text for document {doc_hash}")
            else:
                # Extract text from the image using Gemini Flash
                try:
                    extracted_text = self.extract_text(doc_image)
                except ValueError as e:
                    if 'encoding error' in str(e):
                        # Cut the image resolution in half and try again
                        doc_image = doc_image.resize((doc_image.width // 2, doc_image.height // 2))
                        logger.info(f"Encoding error occurred. Retrying with reduced resolution: {doc_image.size}")
                        try:
                            extracted_text = self.extract_text(doc_image)
                        except ValueError as e:
                            encoding_errors += 1
                    elif 'copyright' in str(e):
                        copyright_errors += 1
                        logger.warning(f"Copyright error for document {doc_hash}")
                    else:
                        encoding_errors += 1
                        logger.error(f"Encoding error for document {doc_hash}: {str(e)}")
                except InternalServerError:
                    server_errors += 1
                    logger.error(f"Internal server error for document {doc_hash}")
                else:
                    with open(cache_file, 'w') as f:
                        json.dump(extracted_text, f, indent=2)
            if 'extracted_text' in locals():
                doc_texts.append(extracted_text)
            else:
                doc_texts.append(None)
        print(f"{len(documents)} OCR'd with {encoding_errors}/{copyright_errors}/{server_errors} encoding/copyright/server errors")

        # not very efficient since we're doing both the embedding and the insert one-at-a-time
        futures = []
        for doc_text, doc_hash in tqdm(zip(doc_texts, document_hashes), desc="Embedding and inserting documents"):
            if doc_text is None or self.db.document_exists(doc_hash):
                continue
            doc_embeddings = get_embeddings(self.model_name, [doc_text], is_query=False)
            futures.append(self.db.insert_documents([(doc_hash, doc_text, doc_embeddings)]))
        
        for future in tqdm(futures, desc="Waiting for inserts to complete"):
            future.result()

        return document_hashes

    def extract_text(self, doc_image):
        response = self.gemini_model.generate_content(
            [
                "Extract all the text from this image, preserving structure as much as possible.",
                doc_image
            ],
            generation_config=genai.types.GenerationConfig(temperature=0, max_output_tokens=2048, )
        )
        extracted_text = response.text
        return extracted_text

    def get_scores(
            self,
            list_emb_queries: list[torch.Tensor],
            list_emb_documents: list[str],
            batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        logger.info(f"Computing scores for {len(list_emb_queries)} queries and {len(list_emb_documents)} documents")

        # Default scores with the correct shape
        # TODO replace this with q query to self.db
        return torch.ones((len(list_emb_queries), len(list_emb_documents)))

    def get_save_one_path(self, output_path, dataset_name):
        fname = f'{self.keyspace_name(dataset_name)}_{self.model_name}.pth'
        return os.path.join(output_path, fname)

    def get_save_all_path(self, output_path):
        return self.get_save_one_path(output_path, 'all')

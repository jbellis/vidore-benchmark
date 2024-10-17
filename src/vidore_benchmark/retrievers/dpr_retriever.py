import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cohere
import google.generativeai as genai
import numpy as np
import tiktoken
import torch
import unstructured_client
from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from cassandra.cluster import Session, Cluster
from colbert_live.db.astra import execute_concurrent_async
from google.api_core.exceptions import InternalServerError
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from more_itertools import chunked
from nltk import word_tokenize
from nltk.corpus import stopwords
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from unstructured_client.models import shared

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device
from .colbert_live_retriever import encode_to_bytes


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


u_client = None
llama_parser = None


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
        print(f'Using keyspace {self.keyspace}')
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
        self.exists_stmt = self.session.prepare("""
            SELECT id FROM documents WHERE id = ?
        """)

    def insert_documents(self, doc_params: list[tuple[str, str, list[float]]]):
        return execute_concurrent_async(self.session, [(self.insert_stmt, params) for params in doc_params])

    def search(self, query_embedding: list[float], k: int) -> list[tuple[str, float]]:
        rows = self.session.execute(self.search_stmt, (query_embedding, query_embedding, k))
        return [(row.id, row.similarity) for row in rows]

    def document_exists(self, doc_id: str) -> bool:
        rows = self.session.execute(self.exists_stmt, (doc_id,))
        return rows.one() is not None


STELLA_MODEL = None
BGE_M3_MODEL = None
openai_client = None

def get_embeddings(provider, texts: list[str], is_query: bool = False) -> list[list[float]]:
    if provider.startswith('openai'):
        global openai_client
        if not openai_client:
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if 'small' in provider:
            model_name = 'text-embedding-3-small'
        else:
            model_name = 'text-embedding-3-large'
        tiktoken_model = tiktoken.encoding_for_model(model_name)
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
        response = openai_client.embeddings.create(input=truncated_texts, model=model_name)
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
    elif provider.startswith('bge-m3'):
        global BGE_M3_MODEL
        if BGE_M3_MODEL is None:
            BGE_M3_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        with torch.no_grad():
            output = BGE_M3_MODEL.encode(texts, max_length=512)["dense_vecs"]
        return output.tolist()
    else:
        raise ValueError(f"Invalid embedding provider: {provider}")


@register_vision_retriever("dpr")
class DprRetriever(VisionRetriever):
    """
    DprSherpaRetriever class to retrieve embeddings using a DPR embeddings model with llm_sherpa
    chunking the images.
    """

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)
        self.query_cache_dir = os.path.join(os.getcwd(), 'query_cache')
        os.makedirs(self.query_cache_dir, exist_ok=True)
        self.embeddings_model = os.environ.get('VIDORE_DPR_EMBEDDINGS')
        self.current_dataset_name = None
        valid_models = ['openai-v3-large', 'openai-v3-small', 'gemini-004', 'stella', 'bge-m3']
        if self.embeddings_model not in valid_models:
            raise ValueError(f"Invalid embeddings model: {self.embeddings_model}. Valid models: {valid_models}")
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')
        self.db = None # initialized by use_dataset
        self.mode = os.environ.get('VIDORE_SCORE_MODE')
        valid_modes = ['bm25', 'dpr', 'reranked']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid scoring mode: {self.mode}. Valid modes: {valid_modes}")

        if self.mode == 'reranked':
            self.cohere_client = cohere.Client(api_key=os.environ.get('COHERE_API_KEY'))
        else:
            self.cohere_client = None

        self.ocr_source = os.environ.get('VIDORE_OCR')
        valid_ocr_sources = ['flash', 'unstructured', 'llamaparse']
        if self.ocr_source not in valid_ocr_sources:
            raise ValueError(f"Invalid OCR source: {self.ocr_source}. Valid sources: {valid_ocr_sources}")

        self.document_cache_dir = os.path.join(os.getcwd(), f'document_cache_{self.ocr_source}')

    def use_dataset(self, ds):
        if 'synthetic' in ds.name:
            raise StopIteration('Reached synthetic datsets; stopping')

        self.current_dataset_name = ds.name
        dataset_cache_dir = os.path.join(self.document_cache_dir, self.current_dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)

        if self.embeddings_model == 'openai-v3-large':
            dim = 1536 * 2
        elif self.embeddings_model == 'openai-v3-small':
            dim = 1536
        elif self.embeddings_model == 'gemini-004':
            dim = 768
        elif self.embeddings_model == 'stella':
            dim = 1024
        elif self.embeddings_model == 'bge-m3':
            dim = 1024
        else:
            raise ValueError(f"Invalid embeddings model: {self.embeddings_model}")
        self.db = DprDB(self.keyspace_name(ds.name), dim)

    def keyspace_name(self, dataset_name):
        return ''.join([c if c.isalnum() else '_' for c in (f'{dataset_name}_{self.ocr_source}_{self.embeddings_model}').lower()])

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: list[str], batch_size: int, **kwargs) -> list[torch.Tensor]:
        batch_size = 32
        logger.info(f"Encoding {len(queries)} queries with batch_size={batch_size}")

        self.query_texts = queries
        encoded_queries = []
        for query in tqdm(queries, desc="Loading cached queries"):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            cache_file = os.path.join(self.query_cache_dir, f"{query_hash}_{self.embeddings_model}.pt")
            if os.path.exists(cache_file):
                encoded_query = torch.load(cache_file)
                encoded_queries.append(encoded_query)
            else:
                encoded_queries.append(None)

        if any(qe is None for qe in encoded_queries):
            missing_queries = [q for q, qe in zip(queries, encoded_queries) if qe is None]
            fresh_encodings = []
            for batch in tqdm(chunked(missing_queries, batch_size), total=len(missing_queries)//batch_size + 1, desc="Encoding queries"):
                batch_encodings = get_embeddings(self.embeddings_model, batch, is_query=True)
                fresh_encodings.extend(batch_encodings)
            
            for q, qe in zip(missing_queries, fresh_encodings):
                query_hash = hashlib.sha256(q.encode()).hexdigest()
                cache_file = os.path.join(self.query_cache_dir, f"{query_hash}_{self.embeddings_model}.pt")
                qe2 = torch.tensor(qe).unsqueeze(0)
                torch.save(qe2, cache_file)
                encoded_queries[queries.index(q)] = qe2

        logger.info(f"Loaded/Encoded {len(encoded_queries)} queries")
        print(len(encoded_queries), 'sample query embedding dimensions', encoded_queries[0].shape)
        return encoded_queries

    def forward_documents(self, documents: list[Image.Image], batch_size: int, **kwargs) -> list[str]:
        batch_size = 16
        with ThreadPoolExecutor() as executor:
            document_bytes = list(tqdm(executor.map(encode_to_bytes, documents), total=len(documents), desc="Encoding to bytes"))

        def compute_sha256(content):
            return hashlib.sha256(content).hexdigest()
        with ThreadPoolExecutor() as executor:
            document_hashes = list(executor.map(compute_sha256, document_bytes))

        logger.info(f"Chunking {len(documents)} documents")

        self.doc_texts = []
        for doc_image, doc_hash in tqdm(zip(documents, document_hashes), total=len(documents), desc="OCR-ing documents"):
            dataset_cache_dir = os.path.join(self.document_cache_dir, self.current_dataset_name)
            cache_file = os.path.join(dataset_cache_dir, f"{doc_hash}.txt")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                logger.info(f"Loaded cached text for document {doc_hash}")
            else:
                # Extract text from the image using Gemini Flash
                if self.ocr_source == 'flash':
                    f = self.ocr_gemini
                elif self.ocr_source == 'unstructured':
                    f = self.ocr_unstructured
                else:
                    assert self.ocr_source == 'llamaparse'
                    f = self.ocr_llama
                extracted_text = f(doc_image, doc_hash)
                if extracted_text is not None:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
            self.doc_texts.append(extracted_text)

        # Batch encoding of documents
        valid_docs = [(doc_text, doc_hash)
                      for doc_text, doc_hash in zip(self.doc_texts, document_hashes)
                      if doc_text is not None and not self.db.document_exists(doc_hash)]
        
        if valid_docs:
            texts_to_encode, hashes_to_encode = zip(*valid_docs)

            encoded_docs = []
            for batch_texts in tqdm(chunked(texts_to_encode, batch_size), total=len(texts_to_encode)//batch_size + 1, desc="Encoding documents"):
                batch_embeddings = get_embeddings(self.embeddings_model, batch_texts, is_query=False)
                encoded_docs.extend(batch_embeddings)
            
            print(f"Inserting {len(encoded_docs)} documents to {self.db.keyspace}")
            
            futures = []
            for doc_text, doc_hash, doc_embedding in zip(texts_to_encode, hashes_to_encode, encoded_docs):
                futures.append(self.db.insert_documents([(doc_hash, doc_text, doc_embedding)]))
            
            for future in tqdm(futures, desc="Waiting for inserts to complete"):
                future.result()

        return document_hashes

    def ocr_gemini(self, doc_image: Image.Image, doc_hash: str) -> str | None:
        try:
            return self._ocr_gemini_once(doc_image)
        except ValueError as e:
            if 'encoding error' in str(e):
                # Cut the image resolution in half and try again
                doc_image = doc_image.resize((doc_image.width // 2, doc_image.height // 2))
                logger.info(f"Encoding error occurred. Retrying with reduced resolution: {doc_image.size}")
                try:
                    return self._ocr_gemini_once(doc_image)
                except ValueError as e:
                    print(f"Encoding failed even at reduced resolution: {doc_image.size}")
            elif 'copyright' in str(e):
                print(f"Copyright error for document {doc_hash}")
            else:
                print(f"Encoding error for document {doc_hash}: {str(e)}")
        except InternalServerError:
            print(f"Internal server error for document {doc_hash}")
        return None

    def _ocr_gemini_once(self, doc_image):
        response = self.gemini_model.generate_content(
            [
                "Extract all the text from this image, preserving structure as much as possible.",
                doc_image
            ],
            generation_config=genai.types.GenerationConfig(temperature=0, max_output_tokens=2048, )
        )
        return response.text

    def ocr_llama(self, doc_image: Image.Image, doc_hash: str) -> str:
        global llama_parser
        if not llama_parser:
            llama_parser = LlamaParse(api_key=os.environ.get('LLAMA_CLOUD_API_KEY'),
                                      result_type="markdown")

        filename = "/tmp/ocr_llama.png"
        doc_image.save(filename)

        file_extractor = {".png": llama_parser}
        L = SimpleDirectoryReader(input_files=[filename], file_extractor=file_extractor).load_data()
        if not L:
            print(f"No text extracted from {doc_hash}")
            return None
        return L[0].get_content()

    def ocr_unstructured(self, doc_image: Image.Image, doc_hash: str) -> str:
        global u_client
        if not u_client:
            u_client = unstructured_client.UnstructuredClient(api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
                                                              server_url=os.getenv("UNSTRUCTURED_API_URL"))

        filename = "/tmp/ocr_unstructured.png"
        doc_image.save(filename)
        if 'tabfquad' in self.current_dataset_name or 'shift' in self.current_dataset_name:
            language = 'fr'
        else:
            language = 'eng'
        req = {
            "partition_parameters": {
                "files": {
                    "content": open(filename, "rb"),
                    "file_name": filename,
                },
                "strategy": shared.Strategy.HI_RES,
                "languages": [language],
            }
        }
        res = u_client.general.partition(request=req)
        return '\n\n'.join(e['text'] for e in res.elements)

    def get_scores(
            self,
            list_emb_queries: list[torch.Tensor],
            list_emb_documents: list[str],
            batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        logger.info(f"Computing scores for {len(list_emb_queries)} queries and {len(list_emb_documents)} documents")

        queries_dict = {idx: query for idx, query in enumerate(self.query_texts)}
        tokenized_queries = self.preprocess_text(queries_dict)
        corpus = {idx: passage for idx, passage in enumerate(self.doc_texts)}
        tokenized_corpus = self.preprocess_text(corpus)
        bm25 = BM25Okapi(tokenized_corpus)

        if self.mode == 'bm25':
            scores = []
            for query in tokenized_queries:
                score = bm25.get_scores(query)
                scores.append(score)
            return torch.tensor(np.array(scores))  # (num_queries, num_docs)
        elif self.mode == 'dpr':
            scores = []
            for query_emb in tqdm(list_emb_queries, desc="Computing scores"):
                query_scores = self.db.search(query_emb[0], 100)
                score_dict = dict(query_scores)
                query_scores = [score_dict.get(doc_id, 0.0) for doc_id in list_emb_documents]
                scores.append(query_scores)
            return torch.tensor(scores)

        assert self.mode == 'reranked'
        final_scores = []
        for query_idx, (query, query_emb) in enumerate(tqdm(zip(tokenized_queries, list_emb_queries), total=len(list_emb_queries), desc="Computing scores")):
            # Get BM25 scores
            bm25_scores = bm25.get_scores(query)
            bm25_top_20 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:20]

            # Get DPR scores
            dpr_scores = self.db.search(query_emb[0].tolist(), 20)
            hash_to_index = {hash: idx for idx, hash in enumerate(list_emb_documents)}
            dpr_scores_indexed = [(hash_to_index[doc_hash], score) for doc_hash, score in dpr_scores]

            # Combine BM25 and DPR results
            combined_results = set(doc_index for doc_index, _ in bm25_top_20 + dpr_scores_indexed)

            # Prepare documents for reranking
            documents_to_rerank = [self.doc_texts[doc_index] for doc_index in combined_results]
            
            # Rerank using Cohere's v3 API
            reranked_results = self.cohere_client.rerank(
                query=self.query_texts[query_idx],
                documents=documents_to_rerank,
                model='rerank-multilingual-v3.0',
                top_n=5
            )
            
            # Create a dictionary of reranked scores
            reranked_scores = {doc_id: 0.0 for doc_id in list_emb_documents}
            for result in reranked_results.results:
                doc_id = list(combined_results)[int(result.index)]
                reranked_scores[list_emb_documents[doc_id]] = result.relevance_score
            
            final_scores.append([reranked_scores[doc_id] for doc_id in list_emb_documents])

        return torch.tensor(final_scores)

    def preprocess_text(self, documents: dict[int, str]) -> list[list[str]]:
        """
        Basic preprocessing of the text data:
        - remove stopwords
        - punctuation
        - lowercase all the words.
        """
        stop_words = set(stopwords.words("english"))
        tokenized_list = [[word.lower() for word in word_tokenize(re.sub(r'\|[-\s|]*\|', ' ', sentence))
                           if word.isalnum() and word.lower() not in stop_words]
                          for sentence in documents.values()]
        return tokenized_list

    def get_save_one_path(self, output_path, dataset_name):
        if self.mode == 'dpr':
            mode_name = self.embeddings_model
        elif self.mode == 'bm25':
            mode_name = 'bm25'
        else:
            assert self.mode == 'reranked'
            mode_name = self.embeddings_model + '_reranked'
        return ''.join([c if c.isalnum() else '_' for c in (f'{dataset_name}_{self.ocr_source}_{mode_name}').lower()])

    def get_save_all_path(self, output_path):
        return self.get_save_one_path(output_path, 'all')

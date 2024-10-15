import hashlib
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List
from typing import Optional

import torch
from PIL import Image
from google.api_core.exceptions import InternalServerError
from tqdm import tqdm
from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

import google.generativeai as genai

from .colbert_live_retriever import encode_to_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def use_dataset(self, ds):
        self.dataset_name = ds.name
        keyspace = self.keyspace_name(ds.name)

    def keyspace_name(self, dataset_name):
        return ''.join([c if c.isalnum() else '_' for c in (dataset_name + '_' + self.model_name).lower()])

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
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

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[str]:
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
        for doc_image, doc_hash in tqdm(zip(documents, document_hashes), total=len(documents), desc="Processing documents"):
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
        print(f"{len(documents)} processed with {encoding_errors}/{copyright_errors}/{server_errors} encoding/copyright/server errors")

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
            list_emb_queries: List[torch.Tensor],
            list_emb_documents: List[str],
            batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        logger.info(f"Computing scores for {len(list_emb_queries)} queries and {len(list_emb_documents)} documents")

        # Default scores with the correct shape
        return torch.ones((len(list_emb_queries), len(list_emb_documents)))

    def get_save_one_path(self, output_path, dataset_name):
        fname = f'{self.keyspace_name(dataset_name)}_{self.model_name}.pth'
        return os.path.join(output_path, fname)

    def get_save_all_path(self, output_path):
        return self.get_save_one_path(output_path, 'all')

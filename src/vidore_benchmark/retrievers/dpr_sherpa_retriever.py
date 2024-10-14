import hashlib
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List
from typing import Optional

import torch
from PIL import Image
from llmsherpa.readers import LayoutPDFReader
from tqdm import tqdm
from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

from .colbert_live_retriever import encode_to_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_to_tmp_pdf(img: Image.Image, hash_str: str) -> str:
    fname = '/tmp/' + hash_str + '.pdf'
    img.save(fname, 'PDF', resolution=100.0)
    return fname


@register_vision_retriever("dpr_sherpa")
class DprSherpaRetriever(VisionRetriever):
    """
    DprSherpaRetriever class to retrieve embeddings using a DPR embeddings model with llm_sherpa
    chunking the images.
    """

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)
        self.query_cache_dir = os.path.join(os.getcwd(), 'query_cache')
        os.makedirs(self.query_cache_dir, exist_ok=True)
        self.document_cache_dir = os.path.join(os.getcwd(), 'document_cache')
        os.makedirs(self.document_cache_dir, exist_ok=True)
        self.model_name = 'openai-v3-small'

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
            document_hashes = list(executor.map(compute_sha256, document_bytes), total=len(document_bytes), desc="Computing SHA256")

        logger.info(f"Chunking {len(documents)} documents")
        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)

        for doc_image, doc_hash in tqdm(zip(documents, document_hashes), total=len(documents), desc="Processing documents"):
            cache_file = os.path.join(self.document_cache_dir, f"{doc_hash}.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    chunks = json.load(f)
                logger.info(f"Loaded cached chunks for document {doc_hash}")
            else:
                fname = save_to_tmp_pdf(doc_image, doc_hash)
                doc = pdf_reader.read_pdf(fname)
                chunks = doc.to_chunks()
                
                with open(cache_file, 'w') as f:
                    json.dump(chunks, f, indent=2)
                logger.info(f"Computed and cached chunks for document {doc_hash}")

            # You can process the chunks here if needed
            # print(chunks)

        return document_hashes

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
        fname = f'{self.keyspace_name(dataset_name)}_{self.query_pool_distance}_{self.n_ann_docs}_{self.n_maxsim_candidates}.pth'
        return os.path.join(output_path, fname)

    def get_save_all_path(self, output_path):
        return self.get_save_one_path(output_path, 'all')

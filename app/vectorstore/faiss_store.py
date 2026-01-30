from typing import List, Dict
import os
import json
import faiss
import numpy as np

from app.config import INDEX_PATH, METADATA_PATH
from app.logger import get_logger

logger = get_logger()

class FaissVectorStore:
    """
        Disk-backed FAISS vector store for cosine similarity search.
        Designed for CPU-only, small-to-medium scale RAG workloads.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS inedx for cosine similarity search.
        Assumes embeddings are L2-normalized.
        """

        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict] = []

        logger.info(f"Initialized FAISS IndexFlatIP  | embedding_dim={embedding_dim}")

        
    def add(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        Add Embeddings and their corresponding metadata to the index.
        """

        if embeddings.ndim != 2:
            raise ValueError("Embedding must be a 2D array")
        
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings count does not match metadata count")
        
        logger.info(f"Adding Embeddings to FAISS | count={embeddings.shape[0]}")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    
    def save(self) -> None:
        """
        Perisits FAISS index and metadata to disk.
        """

        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

        logger.warning("Overwriting existing FAISS index on disk")

        faiss.write_index(self.index, INDEX_PATH)

        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

        logger.info(f"FAISS index saved | total_vectors={self.index.ntotal}")


    @classmethod
    def load(cls) -> "FaissVectorStore":
        """
        Load FAISS index and metadata from disk.
        """

        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("FAISS index file not found")

        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("Metadata file not found")

        index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        store = cls(index.d)
        store.index = index
        store.metadata = metadata

        logger.info(
            f"FAISS index loaded | total_vectors={index.ntotal}"
        )

        return store

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Perform similarity search and return top-k results with scores.
        """

        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty")
        
        if query_embedding.ndim != 2:
            raise ValueError("Query embedding must be 2D")
        
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            item = self.metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)

        logger.info(f"FAISS search completed | returned={len(results)}")

        return results
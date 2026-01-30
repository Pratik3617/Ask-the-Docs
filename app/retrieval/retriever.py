from typing import List, Dict
import numpy as np

from app.config import TOP_K
from app.logger import get_logger
from app.vectorstore.faiss_store import FaissVectorStore
from app.ingestion.embedder import embedding_model

logger = get_logger()

def retrieve_context(
    query: str,
    store: FaissVectorStore,
    top_k: int = TOP_K
) -> List[Dict]:
    """
    Embed the user query and retrieve top-k relevant chunks from FAISS
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if store.index.ntotal == 0:
        raise ValueError("FAISS index is empty")
    
    logger.info(f"Starting retrieval | query_length={len(query)} | top_k={top_k}")

    # Embed query (same embedding space as documents)
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)

    results = store.search(query_embedding, top_k)

    logger.info(
        f"Retrieval completed | retrieved chunks={len(results)}"
    )

    return results
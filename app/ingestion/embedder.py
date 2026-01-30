from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL
from app.logger import get_logger

logger = get_logger()

logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def embed_chunks(chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generated embeddings for document chunks.

    Args:
        chunks : List of chunk dictionaries.

    Returns:
        embeddings: Numpy array of shape(num_chunks, embedding_dim)
        metadata: List of metadata dictionaries aligned with embeddings
    """

    if not chunks:
        raise ValueError("No chunks provided for embedding")
    
    texts = [chunk["text"] for chunk in chunks]

    logger.info(f"Generating Embeddings | num_chunks={len(texts)}")

    embeddings = embedding_model.encode(
        texts,
        batch_size = 32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "start_char_pos": chunk["start_char_pos"],
            "end_char_pos": chunk["end_char_pos"]
        }
        for chunk in chunks
    ]

    logger.info(f"Embeddings generated | shape={embeddings.shape}")

    return embeddings, metadata



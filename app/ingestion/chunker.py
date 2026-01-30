from typing import Dict, List
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.logger import get_logger

logger = get_logger()

def normalize_text(text: str) -> str:
    """
    Normalize whitespace and line breaks for consistent chunking.
    """
    if not text:
        return ""
    
    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r","\n")

    # remove exessive whitespaces and empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    return "\n".join(lines)

def chunk_text(text: str) -> List[Dict]:
    """
    Split normalized text into overlapping chunks with metadata.

    Returns : 
        List of chunks each containing:
        - chunk_id, text, start_char_pos, end_char_pos
    """

    normalize_text = normalize_text(text)
    text_length = len(normalize_text)

    if text_length == 0:
        raise ValueError("Cannot chunk empty text")
    
    logger.info(f"Starting chunking process | text length = {text_length}")

    chunks : List[Dict] = []
    start = 0
    chunk_id = 0

    while start < text_length:
        end = start + CHUNK_SIZE
        chunk_content = normalize_text[start:end]

        if chunk_content.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "text" : chunk_content,
                "start_char_pos": start,
                "end_char_pos": min(end, text_length)
            })

            chunk_id += 1
            start += (CHUNK_SIZE - CHUNK_OVERLAP)

    logger.info(f"chunking completed | total chunks = {len(chunks)}")

    return chunks

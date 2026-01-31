from typing import List, Dict
from app.logger import get_logger

logger = get_logger()

SYSTEM_INSTRUCTIONS = """You are a helpful assistant.

Answer the question using ONLY the context provided below.
If the answer is not present in the context, say "I don't know".
Be concise and factual.
"""

def build_prompt(
    question: str,
    retrieved_chunks: List[Dict]
) -> str:
    """
    Build a grounded prompt from retreived chunks and user question.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not retrieved_chunks:
        logger.warning("No retreived chunks provided")

    context_blocks = []
    for i,item in enumerate(retrieved_chunks, start=1):
        print("item:",item)
        context_blocks.append(
            f"[context {i}\n{item['text']}]"
        )
    
    context = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_INSTRUCTIONS}
        Context: {context}
        Question: {question}
        Answer:
    """

    logger.info("Prompt successfully constructed")

    return prompt

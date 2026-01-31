from app.config import MAX_CONTEXT_TOKENS
from app.utils.tokenizer_utils import count_tokens
from app.retrieval.intent import detect_intent
from app.logger import get_logger

logger = get_logger()


def build_prompt(
    question: str,
    retrieved_chunks: list,
    tokenizer
) -> str:
    """
    Build an intent-aware, token-budgeted prompt.
    """

    intent = detect_intent(question)

    # Instruction templates
    if intent == "summarization":
        instruction = (
            "Summarize the document using ONLY the information in the context below.\n"
            "You may combine and rephrase points from the context, but do not add "
            "any external knowledge or assumptions.\n\n"
        )

    elif intent == "definition":
        instruction = (
            "Answer the question using ONLY the information present in the context.\n"
            "You may rephrase or combine sentences from the context, but do not add "
            "external information.\n\n"
        )

    elif intent == "extractive":
        instruction = (
            "List the main topics explicitly mentioned in the context below.\n"
            "Do not infer or add new topics.\n\n"
        )

    else:  # factual QA
        instruction = (
            "Answer the question using ONLY the context below.\n"
            "If the answer cannot be answered from the context, say \"I don't know\".\n\n"
        )

    # Context packing 

    used_tokens = count_tokens(instruction + question, tokenizer)
    context_blocks = []

    for chunk in retrieved_chunks:
        text = chunk["text"].strip()
        token_count = count_tokens(text, tokenizer)

        if used_tokens + token_count > MAX_CONTEXT_TOKENS:
            logger.info("Context token budget reached; stopping chunk addition")
            break

        context_blocks.append(text)
        used_tokens += token_count

    context = "\n\n".join(context_blocks)

    prompt = (
        f"{instruction}"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )

    logger.info(
        f"Prompt constructed | intent={intent} | context_tokens={used_tokens}"
    )

    return prompt

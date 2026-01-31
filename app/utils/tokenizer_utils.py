def count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens using the same tokenizer as the LLM.
    This MUST match the model tokenizer to avoid truncation.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))

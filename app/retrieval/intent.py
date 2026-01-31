def detect_intent(question: str) -> str:
    """
    Detect the intent of the user question.
    """
    q = question.lower().strip()

    if any(k in q for k in ["summarize", "summary", "overview"]):
        return "summarization"

    if any(k in q for k in ["what is", "define", "explain"]):
        return "definition"

    if any(k in q for k in ["list", "topics", "mentioned"]):
        return "extractive"

    return "qa"

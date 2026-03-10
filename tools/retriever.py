"""Simple placeholder retriever interface for document/ontology access."""


def retrieve(query: str) -> dict:
    """Return a stub retrieval result.

    Args:
        query: User or agent query string.

    Returns:
        Dict with placeholder retrieval metadata.
    """
    return {"query": query, "results": [], "status": "stub"}

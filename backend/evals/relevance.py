def compute_relevance(response: str, query: str) -> dict:
    """BGE cosine similarity between query and response embeddings.

    NLI entailment is the wrong abstraction here (a question rarely entails
    its answer), so we use the same embedding space the retriever uses.
    """
    if not response.strip() or not query.strip():
        return {"relevance": 0.0}

    from backend.retrieval.vector_store import embed_texts

    vecs = embed_texts([query, response])
    return {"relevance": round(max(0.0, float(vecs[0] @ vecs[1])), 4)}

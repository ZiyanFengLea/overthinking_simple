"""
Optional semantic-answer matching.

This file is not used by run_examples.py by default. Use it later for ablation,
e.g. canonical-only vs canonical + semantic fallback.
"""

from typing import Any, Optional

import numpy as np

from answer_utils import clean_for_compare, canonicalize_answer


_embedder = None


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load the embedding model only when semantic matching is used."""
    global _embedder

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)

    return _embedder


def try_parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except ValueError:
        return None


def semantic_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two cleaned answer texts."""
    embedder = get_embedder()

    text_a = f"The answer is: {clean_for_compare(a)}"
    text_b = f"The answer is: {clean_for_compare(b)}"

    embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def answers_equivalent(
    a: Any,
    b: Any,
    semantic_threshold: float = 0.82,
    use_semantic: bool = True,
) -> bool:
    """
    Hybrid answer equivalence.

    1. canonical exact match
    2. numeric equality
    3. semantic embedding fallback for non-numeric textual answers
    """
    ca = canonicalize_answer(a)
    cb = canonicalize_answer(b)

    if ca == cb:
        return True

    fa = try_parse_float(ca)
    fb = try_parse_float(cb)

    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-6

    # Avoid risky semantic comparison when only one side is numeric.
    if fa is not None or fb is not None:
        return False

    if not use_semantic:
        return False

    return semantic_similarity(a, b) >= semantic_threshold

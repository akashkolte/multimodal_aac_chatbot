import re
from threading import Lock, Semaphore

import torch

from backend.config.settings import settings

_model = None
_entail_idx: int | None = None
_model_lock = Lock()
_predict_sem = Semaphore(1)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_SENTENCES = 20
_MIN_SENTENCE_WORDS = 2


def _get_model():
    global _model, _entail_idx
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import CrossEncoder

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        model = CrossEncoder(settings.nli_model, device=device)
        label2id = getattr(model.config, "label2id", None) or {}
        for label, idx in label2id.items():
            if label.lower() == "entailment":
                _entail_idx = int(idx)
                break
        if _entail_idx is None:
            raise RuntimeError(
                f"NLI model {settings.nli_model!r} has no 'entailment' label"
            )
        _model = model
        return _model


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return [s for s in parts if len(s.split()) >= _MIN_SENTENCE_WORDS]


def compute_faithfulness(response: str, chunks: list[dict]) -> dict:
    """Sentence-level NLI: each sentence must be entailed by at least one chunk."""
    if not chunks:
        return {"groundedness": 0.0, "hallucination_rate": 0.0, "no_evidence": True}

    sentences = _split_sentences(response)
    # Too short to score meaningfully (one-word replies, fragments). Flagging as
    # no_evidence is honest: we're not scoring it, so it should be excluded from
    # groundedness averages downstream.
    if not sentences:
        return {"groundedness": 0.0, "hallucination_rate": 0.0, "no_evidence": True}

    chunk_texts = [c.get("text", "") for c in chunks if c.get("text")]
    if not chunk_texts:
        return {"groundedness": 0.0, "hallucination_rate": 0.0, "no_evidence": True}

    if len(sentences) > _MAX_SENTENCES:
        sentences = sentences[:_MAX_SENTENCES]

    model = _get_model()
    # NLI pair order: (premise, hypothesis). Chunks are evidence (premise),
    # response sentences are the claims being checked (hypothesis).
    pairs = [(chunk, sent) for sent in sentences for chunk in chunk_texts]
    with _predict_sem:
        logits = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    entail = probs[:, _entail_idx]

    n_chunks = len(chunk_texts)
    threshold = settings.faithfulness_threshold
    grounded = 0
    for i in range(len(sentences)):
        sent_scores = entail[i * n_chunks : (i + 1) * n_chunks]
        if sent_scores.max() >= threshold:
            grounded += 1

    total = len(sentences)
    groundedness = grounded / total
    return {
        "groundedness": round(groundedness, 4),
        "hallucination_rate": round(1.0 - groundedness, 4),
        "no_evidence": False,
    }

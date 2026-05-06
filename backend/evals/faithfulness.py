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


def _no_evidence_result() -> dict:
    return {
        "groundedness": 0.0,
        "hallucination_rate": 0.0,
        "no_evidence": True,
        "sentences_total": 0,
        "sentences_grounded": 0,
        "nli_threshold": settings.faithfulness_threshold,
    }


def _score_from_entail(entail, sentences: list[str], n_chunks: int) -> dict:
    threshold = settings.faithfulness_threshold
    grounded = 0
    for i in range(len(sentences)):
        sent_scores = entail[i * n_chunks : (i + 1) * n_chunks]
        if sent_scores.max() >= threshold:
            grounded += 1
    total = len(sentences)
    return {
        "groundedness": round(grounded / total, 4),
        "hallucination_rate": round(1.0 - grounded / total, 4),
        "no_evidence": False,
        "sentences_total": total,
        "sentences_grounded": grounded,
        "nli_threshold": threshold,
    }


def compute_faithfulness(response: str, chunks: list[dict]) -> dict:
    """Sentence-level NLI: each sentence must be entailed by at least one chunk."""
    return compute_faithfulness_batch([response], chunks)[0]


def compute_faithfulness_batch(responses: list[str], chunks: list[dict]) -> list[dict]:
    """Score multiple candidate responses against the same chunks in a single
    model.predict call. Caller passes `responses` in candidate order; we return
    results in the same order. Falls back to _no_evidence_result when there's
    nothing to score for a given response."""
    chunk_texts = [c.get("text", "") for c in chunks if c.get("text")] if chunks else []
    if not chunk_texts:
        return [_no_evidence_result() for _ in responses]

    per_resp_sentences = [
        _split_sentences(r)[:_MAX_SENTENCES] if r else [] for r in responses
    ]
    pairs: list[tuple[str, str]] = []
    for sents in per_resp_sentences:
        for sent in sents:
            for chunk in chunk_texts:
                pairs.append((chunk, sent))
    if not pairs:
        return [_no_evidence_result() for _ in responses]

    model = _get_model()
    with _predict_sem:
        logits = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    entail = probs[:, _entail_idx]

    out: list[dict] = []
    cursor = 0
    n_chunks = len(chunk_texts)
    for sentences in per_resp_sentences:
        if not sentences:
            out.append(_no_evidence_result())
            continue
        n_pairs = len(sentences) * n_chunks
        out.append(
            _score_from_entail(entail[cursor : cursor + n_pairs], sentences, n_chunks)
        )
        cursor += n_pairs
    return out

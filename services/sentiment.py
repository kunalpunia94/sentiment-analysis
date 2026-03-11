"""Sentiment analysis utilities for the Multimodal Multilingual Sentiment Intelligence Platform.

This module preserves the original sentiment processing workflow (cleaning, chunking,
analyzing, and aggregating) while adding multi-model support and Streamlit-friendly
caching so the app can remain responsive on CPU-only environments.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import math

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:  # Imported lazily so the module also works in non-Streamlit contexts.
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit not available during some tests.
    st = None  # type: ignore


# ---------------------------------------------------------------------------
# Model registry and configuration
# ---------------------------------------------------------------------------

# Primary lightweight English model (instruction requirement).
DEFAULT_ENGLISH_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Multilingual model that supports a wide range of non-English languages.
MULTILINGUAL_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# Legacy model retained for backwards compatibility with previous versions.
LEGACY_MODEL = "textattack/bert-base-uncased-SST-2"

MODEL_REGISTRY: Dict[str, str] = {
    "english": DEFAULT_ENGLISH_MODEL,
    "multilingual": MULTILINGUAL_MODEL,
    "legacy": LEGACY_MODEL,
}

# Retain original constant name but point to an on-hub resource instead of local path.
MODEL_PATH = DEFAULT_ENGLISH_MODEL


def _cache_resource(func):
    """Wrap a function with Streamlit or LRU caching depending on availability."""

    if st is not None:
        return st.cache_resource(show_spinner=False)(func)

    from functools import lru_cache

    return lru_cache(maxsize=len(MODEL_REGISTRY))(func)


@_cache_resource
def _load_tokenizer_and_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load and cache the tokenizer/model pair for the supplied Hugging Face model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


def create_model(model_path: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Return the cached tokenizer and model for the requested path/name."""
    model_name = MODEL_REGISTRY.get(model_path, model_path)
    return _load_tokenizer_and_model(model_name)


# ---------------------------------------------------------------------------
# Original sentiment-processing pipeline (logic preserved and extended)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase and normalise incoming text for inference."""
    import re

    try:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s;/]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception:  # pragma: no cover - defensive; mirrors original behaviour.
        pass

    return text


def chunk_text(text: str, tokenizer: AutoTokenizer, max_length: int) -> List[str]:
    """Split text into token-length bounded chunks (original behaviour retained)."""
    tokens = tokenizer.tokenize(text)
    chunks: List[List[str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunks.append(tokens[start:end])
        start += max_length
        if start >= len(tokens):
            break
    return [" ".join(chunk) for chunk in chunks]


def analyze_sentiment(text: str, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification) -> List[np.ndarray]:
    """Return per-chunk probability distributions for the supplied text."""
    chunks = chunk_text(text, tokenizer, max_length=512)
    scores: List[np.ndarray] = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        scores.append(probs[0])

    return scores


def _map_multilingual_score(score: np.ndarray) -> float:
    """Convert 5-class star ratings to the signed format expected downstream."""
    star_indices = np.arange(1, score.size + 1)
    weighted_rating = float(np.dot(score, star_indices))
    midpoint = (score.size + 1) / 2.0
    if score.size == 1:
        return 0.0
    return (weighted_rating - midpoint) / ((score.size - 1) / 2.0)


def aggregate_sentiment_with_sign(scores: Iterable[np.ndarray]) -> Tuple[float, int]:
    """Aggregate chunk scores into a mean signed score and final label."""
    signed_scores: List[float] = []

    for score in scores:
        if score.size == 2:
            sentiment_sign = -1 if int(np.argmax(score)) == 0 else 1
            signed_scores.append(sentiment_sign * float(np.max(score)))
        else:
            signed_scores.append(_map_multilingual_score(score))

    mean_signed_score = float(np.mean(signed_scores)) if signed_scores else 0.0
    final_label = 1 if mean_signed_score > 0 else 0
    return mean_signed_score, final_label


def sentiment_score_calculation(text: str, model_name: Optional[str] = None) -> Tuple[float, int, List[np.ndarray]]:
    """Run the core sentiment pipeline on text using the requested model."""
    cleaned_text = clean_text(text)
    model_key = model_name or MODEL_PATH
    tokenizer, model = create_model(model_key)
    scores = analyze_sentiment(cleaned_text, tokenizer, model)
    final_score, final_label = aggregate_sentiment_with_sign(scores)
    return round(final_score, 4), final_label, scores


def display_sentiment(sentiment_label: int) -> str:
    """Return a friendly label for downstream UI presentation."""
    return "Positive Sentiment" if sentiment_label == 1 else "Negative Sentiment"


def final_output_sentiment_score(text: str, model_name: Optional[str] = None) -> Tuple[str, float, List[np.ndarray]]:
    """Convenience wrapper matching the original public API."""
    final_score, sentiment_label, scores = sentiment_score_calculation(text, model_name=model_name)
    display_message = display_sentiment(sentiment_label)
    return display_message, final_score, scores


# ---------------------------------------------------------------------------
# Helper utilities for model selection and confidence summarisation
# ---------------------------------------------------------------------------

def select_model_for_language(language: str, translate_to_english: bool, analyze_original_language: bool) -> str:
    """Determine which registered model should process the request."""
    language = (language or "").lower()
    if translate_to_english or language in {"en", "eng", "english", ""}:
        return "english"
    if analyze_original_language:
        return "multilingual"
    return "english"


def compute_confidence(scores: Iterable[np.ndarray], final_label: int) -> float:
    """Derive a confidence score from chunk-level probabilities."""
    confidences: List[float] = []
    for score in scores:
        if score.size == 2:
            confidences.append(float(score[final_label]))
        else:
            if final_label == 1:
                positive_prob = float(np.sum(score[math.ceil(score.size / 2) :]))
                confidences.append(positive_prob)
            else:
                negative_prob = float(np.sum(score[: math.floor(score.size / 2)]))
                confidences.append(negative_prob)
    if not confidences:
        return 0.5
    return float(np.mean(confidences))


__all__ = [
    "MODEL_PATH",
    "MODEL_REGISTRY",
    "clean_text",
    "chunk_text",
    "analyze_sentiment",
    "aggregate_sentiment_with_sign",
    "sentiment_score_calculation",
    "final_output_sentiment_score",
    "display_sentiment",
    "select_model_for_language",
    "compute_confidence",
]

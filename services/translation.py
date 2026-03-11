"""Translation helpers powered by lightweight Helsinki-NLP models."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows running translation logic offline.
    st = None  # type: ignore


LanguagePair = Tuple[str, str]


TRANSLATION_MODEL_MAP: Dict[LanguagePair, str] = {
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("te", "en"): "Helsinki-NLP/opus-mt-te-en",
    ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
    ("kn", "en"): "Helsinki-NLP/opus-mt-kn-en",
    ("ml", "en"): "Helsinki-NLP/opus-mt-ml-en",
    ("mr", "en"): "Helsinki-NLP/opus-mt-mr-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("en", "te"): "Helsinki-NLP/opus-mt-en-te",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-ta",
    ("en", "kn"): "Helsinki-NLP/opus-mt-en-kn",
    ("en", "ml"): "Helsinki-NLP/opus-mt-en-ml",
    ("en", "mr"): "Helsinki-NLP/opus-mt-en-mr",
}


def _cache_translation(func):
    if st is not None:
        return st.cache_resource(show_spinner=False)(func)

    from functools import lru_cache

    return lru_cache(maxsize=len(TRANSLATION_MODEL_MAP) or None)(func)


@_cache_translation
def _load_translation_pipeline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer, device=-1)


def is_translation_supported(source_language: Optional[str], target_language: Optional[str]) -> bool:
    if not source_language or not target_language:
        return False
    if source_language == target_language:
        return True
    return (source_language, target_language) in TRANSLATION_MODEL_MAP


def translate_text(text: str, source_language: Optional[str], target_language: Optional[str]) -> Optional[str]:
    if not text or not text.strip():
        return text
    if not source_language or not target_language:
        return None
    if source_language == target_language:
        return text
    model_name = TRANSLATION_MODEL_MAP.get((source_language, target_language))
    if not model_name:
        return None
    translator = _load_translation_pipeline(model_name)
    max_chunk = 500
    segments = [text[i : i + max_chunk] for i in range(0, len(text), max_chunk)] or [text]
    outputs = translator(segments)
    return " ".join(item["translation_text"].strip() for item in outputs)


def translate_to_english(text: str, source_language: Optional[str]) -> Optional[str]:
    return translate_text(text, source_language, "en")


__all__ = [
    "TRANSLATION_MODEL_MAP",
    "is_translation_supported",
    "translate_text",
    "translate_to_english",
]

"""Speech-to-text services using Whisper models for CPU inference."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import whisper

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit not available in some environments.
    st = None  # type: ignore

from .video_audio import convert_audio_to_wav, save_bytes_to_temp, save_uploaded_file

WHISPER_MODEL_SIZE = "small"


def _cache_whisper(func):
    if st is not None:
        return st.cache_resource(show_spinner=False)(func)

    from functools import lru_cache

    return lru_cache(maxsize=1)(func)


@_cache_whisper
def load_whisper_model(model_size: str = WHISPER_MODEL_SIZE):
    return whisper.load_model(model_size, device="cpu")


def transcribe_audio_file(audio_path: Path, language: Optional[str] = None) -> dict:
    model = load_whisper_model()
    wav_path = convert_audio_to_wav(audio_path)
    try:
        result = model.transcribe(str(wav_path), language=language, task="transcribe", fp16=False)
    finally:
        wav_path.unlink(missing_ok=True)
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language"),
        "segments": result.get("segments", []),
    }


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".wav", language: Optional[str] = None) -> dict:
    temp_path = save_bytes_to_temp(audio_bytes, suffix)
    try:
        return transcribe_audio_file(temp_path, language=language)
    finally:
        temp_path.unlink(missing_ok=True)


def transcribe_uploaded_file(uploaded_file, language: Optional[str] = None) -> dict:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    temp_path = save_uploaded_file(uploaded_file, suffix)
    try:
        return transcribe_audio_file(temp_path, language=language)
    finally:
        temp_path.unlink(missing_ok=True)


__all__ = [
    "load_whisper_model",
    "transcribe_audio_file",
    "transcribe_audio_bytes",
    "transcribe_uploaded_file",
]

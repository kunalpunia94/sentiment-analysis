"""Language detection utilities for text and audio workflows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42  # Deterministic results for reproducibility.


@dataclass(frozen=True)
class LanguageOption:
    label: str
    code: str


LANGUAGE_OPTIONS = [
    LanguageOption("Auto-detect", "auto"),
    LanguageOption("English", "en"),
    LanguageOption("Hindi", "hi"),
    LanguageOption("Telugu", "te"),
    LanguageOption("Tamil", "ta"),
    LanguageOption("Kannada", "kn"),
    LanguageOption("Malayalam", "ml"),
    LanguageOption("Marathi", "mr"),
]

LANGUAGE_LABEL_TO_CODE = {option.label: option.code for option in LANGUAGE_OPTIONS}


def detect_text_language(text: str) -> Optional[str]:
    """Run language detection on free-form text using langdetect."""
    try:
        return detect(text)
    except LangDetectException:
        return None


def resolve_language_choice(user_selection: str, auto_detected: Optional[str]) -> Optional[str]:
    """Return the language code respecting manual overrides when provided."""
    selected_code = LANGUAGE_LABEL_TO_CODE.get(user_selection, "auto")
    if selected_code == "auto":
        return auto_detected
    return selected_code


__all__ = [
    "LANGUAGE_OPTIONS",
    "LANGUAGE_LABEL_TO_CODE",
    "detect_text_language",
    "resolve_language_choice",
]

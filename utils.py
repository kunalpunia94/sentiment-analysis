"""Compatibility layer exposing the legacy sentiment helpers.

The full implementation now lives in services.sentiment, but this module keeps the
public surface unchanged for any downstream imports that still reference utils.
"""

from services.sentiment import (  # noqa: F401 - re-exported for backwards compatibility
    MODEL_PATH,
    aggregate_sentiment_with_sign,
    analyze_sentiment,
    clean_text,
    create_model,
    display_sentiment,
    final_output_sentiment_score,
    sentiment_score_calculation,
    chunk_text,
)
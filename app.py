from pathlib import Path
from typing import Optional

import hashlib

import streamlit as st
from audio_recorder_streamlit import audio_recorder

from services.language_detect import LANGUAGE_OPTIONS, detect_text_language, resolve_language_choice
from services.sentiment import (
    compute_confidence,
    display_sentiment,
    select_model_for_language,
    sentiment_score_calculation,
)
from services.speech_to_text import transcribe_audio_bytes, transcribe_uploaded_file, transcribe_audio_file
from services.translation import is_translation_supported, translate_text
from services.video_audio import FFmpegError, extract_audio_from_video, save_uploaded_file


st.set_page_config(page_title="Multimodal Multilingual Sentiment Intelligence Platform", layout="wide")


LANGUAGE_LABEL_TO_CODE = {option.label: option.code for option in LANGUAGE_OPTIONS}
CODE_TO_LANGUAGE_LABEL = {option.code: option.label for option in LANGUAGE_OPTIONS}
NON_AUTO_LANGUAGE_LABELS = [label for label, code in LANGUAGE_LABEL_TO_CODE.items() if code != "auto"]
ENGLISH_CODES = {"en"}
AUTO_TRANSLATE_LABEL = "Auto (use detected)"
RESET_STATE_KEYS = [
    "analysis",
    "live_preview_token",
    "live_preview_data",
    "live_preview_error",
    "audio_preview_token",
    "audio_preview_data",
    "audio_preview_error",
    "prev_input_mode",
]


def reset_app_state() -> None:
    for key in RESET_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.analysis = None


def get_language_code(selection: str) -> str:
    if selection == AUTO_TRANSLATE_LABEL:
        return "auto"
    return LANGUAGE_LABEL_TO_CODE.get(selection, "auto")


def language_label(code: Optional[str]) -> str:
    if not code:
        return "Unknown"
    return CODE_TO_LANGUAGE_LABEL.get(code, code)


def run_sentiment(
    text: str,
    original_language_code: Optional[str],
    analysis_language_code: Optional[str],
    translation_applied: bool,
    sentiment_choice: str,
):
    language_for_model = analysis_language_code or original_language_code or "en"
    translate_flag = translation_applied or (language_for_model in ENGLISH_CODES)
    analyze_original_flag = not translate_flag and language_for_model not in ENGLISH_CODES

    if sentiment_choice == "English":
        if language_for_model not in ENGLISH_CODES:
            st.warning("English sentiment selected while text is not English. Enable translation for optimal accuracy.")
        model_key = "english"
    elif sentiment_choice == "Original language":
        model_key = "multilingual"
    else:
        model_key = select_model_for_language(language_for_model, translate_flag, analyze_original_flag)

    final_score, label, scores = sentiment_score_calculation(text, model_key)
    message = display_sentiment(label)
    confidence = compute_confidence(scores, label)
    return {
        "score": round(final_score, 2),
        "label": label,
        "message": message,
        "confidence": round(confidence, 4),
        "model_key": model_key,
    }


if "analysis" not in st.session_state:
    st.session_state.analysis = None


st.title("Multimodal Multilingual Sentiment Intelligence Platform")
st.caption("Analyze sentiment across text, speech, audio, and video using a unified pipeline.")

sidebar = st.sidebar
sidebar.header("Configuration")

if sidebar.button("Reset Tool", type="secondary"):
    reset_app_state()
    st.rerun()

input_mode = sidebar.radio(
    "Input Type",
    options=["Text", "Live Speech", "Audio Upload", "Video Upload"],
)

if "prev_input_mode" not in st.session_state:
    st.session_state.prev_input_mode = input_mode
elif st.session_state.prev_input_mode != input_mode:
    reset_app_state()
    st.session_state.prev_input_mode = input_mode

language_selection = sidebar.selectbox(
    "Input language",
    [option.label for option in LANGUAGE_OPTIONS],
)

sentiment_model_choice = sidebar.selectbox(
    "Sentiment model language",
    ["Auto (based on pipeline)", "English", "Original language"],
)

translate_toggle = sidebar.checkbox("Enable translation", value=True)

translation_source_selection = None
translation_target_selection = None

if translate_toggle:
    translation_source_selection = sidebar.selectbox(
        "Translate from",
        [AUTO_TRANSLATE_LABEL] + NON_AUTO_LANGUAGE_LABELS,
    )
    default_target_index = NON_AUTO_LANGUAGE_LABELS.index("English") if "English" in NON_AUTO_LANGUAGE_LABELS else 0
    translation_target_selection = sidebar.selectbox(
        "Translate to",
        NON_AUTO_LANGUAGE_LABELS,
        index=default_target_index,
    )

language_override_code = get_language_code(language_selection)
translation_source_code = get_language_code(translation_source_selection) if translation_source_selection else None
translation_target_code = get_language_code(translation_target_selection) if translation_target_selection else None
if translation_target_code == "auto":
    translation_target_code = None

translation_settings = {
    "enabled": translate_toggle,
    "source_code": translation_source_code,
    "target_code": translation_target_code,
}


def maybe_translate_text(
    text: str,
    resolved_language: Optional[str],
    settings: dict,
):
    analysis_language = resolved_language
    translation_applied = False
    translation_text = None
    translation_target = None
    translation_source = resolved_language

    if not settings.get("enabled"):
        return translation_text, text, analysis_language, translation_applied, translation_target, translation_source

    if not text or not text.strip():
        return translation_text, text, analysis_language, translation_applied, translation_target, translation_source

    source_override = settings.get("source_code")
    target_code = settings.get("target_code")

    effective_source = resolved_language
    if source_override and source_override != "auto":
        effective_source = source_override

    if not effective_source:
        st.warning("Unable to determine source language for translation.")
        return translation_text, text, analysis_language, translation_applied, translation_target, translation_source

    if not target_code:
        st.warning("Select a valid translation target language.")
        return translation_text, text, effective_source, translation_applied, translation_target, effective_source

    if effective_source == target_code:
        return translation_text, text, effective_source, translation_applied, target_code, effective_source

    if not is_translation_supported(effective_source, target_code):
        st.warning("Translation unavailable for the selected language pair.")
        return translation_text, text, effective_source, translation_applied, translation_target, effective_source

    translated = translate_text(text, effective_source, target_code)
    if not translated:
        st.warning("Translation failed; using original text.")
        return translation_text, text, effective_source, translation_applied, translation_target, effective_source

    translation_text = translated
    analysis_language = target_code
    translation_applied = True
    translation_target = target_code
    translation_source = effective_source
    return translation_text, translation_text, analysis_language, translation_applied, translation_target, translation_source

text_input = ""
recorded_audio = None
audio_file = None
video_file = None

if input_mode == "Text":
    text_input = st.text_area("Enter or paste text", height=220, max_chars=10000)
elif input_mode == "Live Speech":
    st.markdown("#### Record live speech")
    st.info("Click to start recording, speak, and click again to stop. Audio stays on device until you submit.")
    recorded_audio = audio_recorder(pause_threshold=2.0, sample_rate=16000)
    whisper_language = None if language_override_code == "auto" else language_override_code
    if recorded_audio:
        st.audio(recorded_audio, format="audio/wav")
        audio_hash = hashlib.sha256(recorded_audio).hexdigest()
        if st.session_state.get("live_preview_token") != audio_hash:
            try:
                with st.spinner("Transcribing live speech preview..."):
                    preview_tx = transcribe_audio_bytes(
                        recorded_audio,
                        suffix=".wav",
                        language=whisper_language,
                    )
            except FFmpegError as exc:
                st.session_state["live_preview_error"] = str(exc)
                st.session_state["live_preview_data"] = None
            else:
                st.session_state["live_preview_token"] = audio_hash
                st.session_state["live_preview_data"] = preview_tx
                st.session_state["live_preview_error"] = None
        if st.session_state.get("live_preview_error"):
            st.error(f"Live transcription failed: {st.session_state['live_preview_error']}")
        elif st.session_state.get("live_preview_data"):
            preview = st.session_state["live_preview_data"]
            preview_text = preview.get("text", "").strip()
            st.text_area(
                "Live transcript preview",
                preview_text,
                height=160,
                disabled=True,
            )
            detected_code = preview.get("language")
            if detected_code:
                st.caption(f"Detected language: {CODE_TO_LANGUAGE_LABEL.get(detected_code, detected_code)}")
elif input_mode == "Audio Upload":
    audio_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
    )
    if audio_file is not None:
        st.audio(audio_file)
        whisper_language = None if language_override_code == "auto" else language_override_code
        file_token = f"{audio_file.name}:{getattr(audio_file, 'size', None)}"
        if st.session_state.get("audio_preview_token") != file_token:
            try:
                with st.spinner("Transcribing audio preview..."):
                    preview_tx = transcribe_uploaded_file(audio_file, language=whisper_language)
            except FFmpegError as exc:
                st.session_state["audio_preview_error"] = str(exc)
                st.session_state["audio_preview_data"] = None
            else:
                st.session_state["audio_preview_token"] = file_token
                st.session_state["audio_preview_data"] = preview_tx
                st.session_state["audio_preview_error"] = None
            finally:
                audio_file.seek(0)
        if st.session_state.get("audio_preview_error"):
            st.error(f"Audio transcription failed: {st.session_state['audio_preview_error']}")
        elif st.session_state.get("audio_preview_data"):
            preview = st.session_state["audio_preview_data"]
            preview_text = preview.get("text", "").strip()
            st.text_area(
                "Uploaded audio transcript",
                preview_text,
                height=160,
                disabled=True,
            )
            detected_code = preview.get("language")
            if detected_code:
                st.caption(f"Detected language: {CODE_TO_LANGUAGE_LABEL.get(detected_code, detected_code)}")
elif input_mode == "Video Upload":
    video_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "avi", "mkv"],
    )


def handle_text_input(raw_text: str) -> Optional[dict]:
    if not raw_text or not raw_text.strip():
        st.error("Please provide text for analysis.")
        return None
    auto_detected = None if language_override_code != "auto" else detect_text_language(raw_text)
    resolved_language = resolve_language_choice(language_selection, auto_detected)
    (
        translation_text,
        processed_text,
        analysis_language,
        translation_applied,
        translation_target,
        translation_source,
    ) = maybe_translate_text(raw_text, resolved_language, translation_settings)
    result = run_sentiment(
        processed_text,
        resolved_language,
        analysis_language,
        translation_applied,
        sentiment_model_choice,
    )
    st.session_state.analysis = {
        "input_mode": "Text",
        "original_text": raw_text,
        "transcribed_text": raw_text,
        "translated_text": translation_text,
        "detected_language": auto_detected,
        "resolved_language": resolved_language,
        "analysis_language": analysis_language,
        "translation_target": translation_target,
        "translation_source": translation_source,
        "translation_applied": translation_applied,
        **result,
    }
    return st.session_state.analysis


def handle_audio_bytes(
    data: bytes,
    suffix: str,
    cached_transcription: Optional[dict] = None,
) -> Optional[dict]:
    if not data:
        st.error("No audio detected. Record or upload audio before running analysis.")
        return None
    whisper_language = None if language_override_code == "auto" else language_override_code
    if cached_transcription and cached_transcription.get("text"):
        transcription = cached_transcription
    else:
        try:
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio_bytes(data, suffix=suffix, language=whisper_language)
        except FFmpegError as exc:
            st.error(f"FFmpeg error: {exc}")
            return None
    transcript = transcription.get("text", "").strip()
    if not transcript:
        st.error("Transcription produced empty text. Try recording again or use clearer audio.")
        return None
    auto_detected = transcription.get("language") if language_override_code == "auto" else None
    resolved_language = resolve_language_choice(language_selection, auto_detected)
    (
        translation_text,
        processed_text,
        analysis_language,
        translation_applied,
        translation_target,
        translation_source,
    ) = maybe_translate_text(transcript, resolved_language, translation_settings)
    result = run_sentiment(
        processed_text,
        resolved_language,
        analysis_language,
        translation_applied,
        sentiment_model_choice,
    )
    st.session_state.analysis = {
        "input_mode": input_mode,
        "original_text": transcript,
        "transcribed_text": transcript,
        "translated_text": translation_text,
        "detected_language": auto_detected,
        "resolved_language": resolved_language,
        "analysis_language": analysis_language,
        "translation_target": translation_target,
        "translation_source": translation_source,
        "translation_applied": translation_applied,
        "transcription_details": transcription.get("segments", []),
        **result,
    }
    return st.session_state.analysis


def handle_audio_upload(uploaded, cached_transcription: Optional[dict] = None) -> Optional[dict]:
    if uploaded is None:
        st.error("Please upload an audio file before running analysis.")
        return None
    whisper_language = None if language_override_code == "auto" else language_override_code
    if cached_transcription and cached_transcription.get("text"):
        transcription = cached_transcription
    else:
        try:
            with st.spinner("Transcribing audio file..."):
                transcription = transcribe_uploaded_file(uploaded, language=whisper_language)
        except FFmpegError as exc:
            st.error(f"FFmpeg error: {exc}")
            return None
        finally:
            uploaded.seek(0)
    transcript = transcription.get("text", "").strip()
    if not transcript:
        st.error("Transcription failed. Ensure the audio is clear and try again.")
        return None
    auto_detected = transcription.get("language") if language_override_code == "auto" else None
    resolved_language = resolve_language_choice(language_selection, auto_detected)
    (
        translation_text,
        processed_text,
        analysis_language,
        translation_applied,
        translation_target,
        translation_source,
    ) = maybe_translate_text(transcript, resolved_language, translation_settings)
    result = run_sentiment(
        processed_text,
        resolved_language,
        analysis_language,
        translation_applied,
        sentiment_model_choice,
    )
    st.session_state.analysis = {
        "input_mode": "Audio Upload",
        "original_text": transcript,
        "transcribed_text": transcript,
        "translated_text": translation_text,
        "detected_language": auto_detected,
        "resolved_language": resolved_language,
        "analysis_language": analysis_language,
        "translation_target": translation_target,
        "translation_source": translation_source,
        "translation_applied": translation_applied,
        "transcription_details": transcription.get("segments", []),
        **result,
    }
    return st.session_state.analysis


def handle_video_upload(uploaded) -> Optional[dict]:
    if uploaded is None:
        st.error("Please upload a video file before running analysis.")
        return None
    suffix = Path(uploaded.name).suffix or ".mp4"
    video_path = save_uploaded_file(uploaded, suffix)
    audio_path = None
    try:
        with st.spinner("Extracting audio from video..."):
            audio_path = extract_audio_from_video(video_path)
        whisper_language = None if language_override_code == "auto" else language_override_code
        with st.spinner("Transcribing video audio..."):
            transcription = transcribe_audio_file(audio_path, language=whisper_language)
    except FFmpegError as exc:
        st.error(f"FFmpeg error: {exc}")
        return None
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass
        video_path.unlink(missing_ok=True)

    transcript = transcription.get("text", "").strip()
    if not transcript:
        st.error("Transcription from video resulted in empty text.")
        return None
    auto_detected = transcription.get("language") if language_override_code == "auto" else None
    resolved_language = resolve_language_choice(language_selection, auto_detected)
    (
        translation_text,
        processed_text,
        analysis_language,
        translation_applied,
        translation_target,
        translation_source,
    ) = maybe_translate_text(transcript, resolved_language, translation_settings)
    result = run_sentiment(
        processed_text,
        resolved_language,
        analysis_language,
        translation_applied,
        sentiment_model_choice,
    )
    st.session_state.analysis = {
        "input_mode": "Video Upload",
        "original_text": transcript,
        "transcribed_text": transcript,
        "translated_text": translation_text,
        "detected_language": auto_detected,
        "resolved_language": resolved_language,
        "analysis_language": analysis_language,
        "translation_target": translation_target,
        "translation_source": translation_source,
        "translation_applied": translation_applied,
        "transcription_details": transcription.get("segments", []),
        **result,
    }
    return st.session_state.analysis


if st.button("Analyze Sentiment", type="primary"):
    st.session_state.analysis = None
    if input_mode == "Text":
        handle_text_input(text_input)
    elif input_mode == "Live Speech":
        handle_audio_bytes(
            recorded_audio,
            suffix=".wav",
            cached_transcription=st.session_state.get("live_preview_data"),
        )
    elif input_mode == "Audio Upload":
        if audio_file is not None:
            audio_file.seek(0)
        handle_audio_upload(
            audio_file,
            cached_transcription=st.session_state.get("audio_preview_data"),
        )
    else:
        handle_video_upload(video_file)


analysis = st.session_state.analysis
if analysis:
    st.markdown("---")
    st.subheader("Sentiment Results")
    st.metric("Sentiment", analysis["message"], delta=f"Score: {analysis['score']}")
    st.write(f"Confidence: **{analysis['confidence'] * 100:.1f}%**")
    st.write(f"Model used: **{analysis['model_key']}**")

    meta_parts = []
    if analysis.get("detected_language"):
        meta_parts.append(f"Detected: {language_label(analysis['detected_language'])}")
    if analysis.get("resolved_language"):
        meta_parts.append(f"Source: {language_label(analysis['resolved_language'])}")
    if analysis.get("analysis_language"):
        meta_parts.append(f"Analyzed as: {language_label(analysis['analysis_language'])}")
    if analysis.get("translation_applied") and analysis.get("translation_target"):
        meta_parts.append(f"Translated to: {language_label(analysis['translation_target'])}")
    if meta_parts:
        st.caption(" | ".join(meta_parts))

    with st.expander("Transcribed Text", expanded=True):
        st.write(analysis["transcribed_text"])

    if analysis.get("translated_text"):
        target_label = language_label(analysis.get("translation_target"))
        with st.expander(f"Translated Text ({target_label})"):
            st.write(analysis["translated_text"])

    st.success("Analysis complete. Adjust configuration or input to run another evaluation.")




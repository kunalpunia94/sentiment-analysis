"""Audio/video preprocessing utilities for Whisper transcription."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import os

AUDIO_SAMPLE_RATE = 16_000


class FFmpegError(RuntimeError):
    """Raised when FFmpeg fails to process an input file."""


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FFmpegError("ffmpeg executable not found. Ensure packages.txt installs ffmpeg.")


def save_bytes_to_temp(data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def save_uploaded_file(uploaded_file, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def _run_ffmpeg(args: list) -> Path:
    ensure_ffmpeg_available()
    output_path = Path(args[-1])
    command = ["ffmpeg", "-y"] + args
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise FFmpegError(process.stderr.decode("utf-8", errors="ignore"))
    return output_path


def convert_audio_to_wav(input_path: Path, output_path: Optional[Path] = None, sample_rate: int = AUDIO_SAMPLE_RATE) -> Path:
    if output_path is None:
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        output_path = Path(temp_path)
    args = ["-i", str(input_path), "-ar", str(sample_rate), "-ac", "1", str(output_path)]
    return _run_ffmpeg(args)


def extract_audio_from_video(video_path: Path, output_path: Optional[Path] = None, sample_rate: int = AUDIO_SAMPLE_RATE) -> Path:
    if output_path is None:
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        output_path = Path(temp_path)
    args = ["-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", str(output_path)]
    return _run_ffmpeg(args)


__all__ = [
    "AUDIO_SAMPLE_RATE",
    "FFmpegError",
    "ensure_ffmpeg_available",
    "save_bytes_to_temp",
    "save_uploaded_file",
    "convert_audio_to_wav",
    "extract_audio_from_video",
]

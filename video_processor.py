"""FFmpeg/FFprobe helpers: probing, vertical cropping, subtitle burning, and rendering."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class VideoError(Exception):
    """Raised when FFmpeg/FFprobe is missing or rendering fails."""


def check_ffmpeg() -> None:
    """Raise VideoError if ffmpeg or ffprobe is not available."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise VideoError(
                f"'{tool}' was not found on PATH. Install FFmpeg and restart the application. "
                "On Windows: winget install Gyan.FFmpeg"
            )


def probe_duration(path: str) -> float:
    """Return the media duration in seconds using ffprobe."""
    check_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VideoError("ffprobe is not installed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoError("ffprobe timed out while reading the video.") from exc

    if result.returncode != 0:
        raise VideoError(f"ffprobe could not read the video: {result.stderr.strip()}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise VideoError("ffprobe returned an invalid duration.") from exc

    if duration <= 0:
        raise VideoError("The video has zero duration.")

    return duration


def _escape_subtitle_path(path: str) -> str:
    """Escape a filesystem path for FFmpeg's subtitles= filter on Windows.

    - Convert backslashes to forward slashes.
    - Escape the drive-letter colon.
    - Escape single quotes.
    - Wrap the path in single quotes.
    """
    p = str(path).replace("\\", "/")
    p = p.replace(":", "\\:")
    p = p.replace("'", "\\'")
    return f"'{p}'"


def render_clip(
    input_video: str,
    clip_srt: str,
    start_seconds: float,
    output_path: str,
    duration: float = 60.0,
) -> str:
    """Render one vertical Short with burned-in captions.

    Returns the output path. Raises VideoError on failure.
    """
    check_ffmpeg()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    escaped_srt = _escape_subtitle_path(clip_srt)

    # Center-crop to 9:16, scale to 1080x1920, then burn captions.
    # Font: bold white text, thick black outline, bottom-centre.
    video_filter = (
        "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"
        "scale=1080:1920:flags=lanczos,"
        f"subtitles={escaped_srt}:"
        "force_style='FontName=Arial,FontSize=22,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "Outline=3,Shadow=1,Bold=1,Alignment=2,MarginV=80'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-i",
        input_video,
        "-t",
        f"{duration:.3f}",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "make_zero",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VideoError("ffmpeg is not installed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoError("Rendering timed out after 15 minutes.") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[-1500:]
        raise VideoError(f"FFmpeg rendering failed: {stderr}")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise VideoError("FFmpeg reported success but the output file is missing or empty.")

    # Validate duration tolerance relative to the requested clip length.
    rendered_duration = probe_duration(output_path)
    tolerance = max(1.5, duration * 0.04)
    if not (duration - tolerance <= rendered_duration <= duration + tolerance):
        raise VideoError(
            f"Rendered clip duration is {rendered_duration:.2f}s, "
            f"outside the expected {duration:.1f}s ± {tolerance:.1f}s window."
        )

    return output_path
"""SRT parsing, timestamp helpers, and clip-local subtitle generation."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Iterable

import srt


class SubtitleError(Exception):
    """Raised when an SRT file cannot be parsed or is unusable."""


def parse_srt_file(path: str) -> list[dict]:
    """Parse an SRT file into a list of cue dictionaries.

    Each cue has:
        index: int
        start: float  (seconds)
        end:   float  (seconds)
        text:  str
    """
    if not os.path.isfile(path):
        raise SubtitleError(f"SRT file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            raw = fh.read()
    except OSError as exc:
        raise SubtitleError(f"Could not read SRT file: {exc}") from exc

    if not raw.strip():
        raise SubtitleError("The SRT file is empty.")

    try:
        subtitles = list(srt.parse(raw))
    except Exception as exc:
        raise SubtitleError(f"Invalid SRT format: {exc}") from exc

    if not subtitles:
        raise SubtitleError("The SRT file contains no subtitle cues.")

    cues: list[dict] = []
    for item in subtitles:
        start = item.start.total_seconds()
        end = item.end.total_seconds()
        if end <= start:
            continue
        text = " ".join(item.content.split())
        if not text:
            continue
        cues.append(
            {
                "index": int(item.index) if item.index is not None else len(cues) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

    if not cues:
        raise SubtitleError("The SRT file contains no usable subtitle cues.")

    cues.sort(key=lambda c: c["start"])
    return cues


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_timestamp_short(seconds: float) -> str:
    """Format seconds as HH:MM:SS for UI display."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def nearest_cue_start(cues: Iterable[dict], target: float, max_distance: float = 2.0) -> float:
    """Return the nearest cue start within max_distance, otherwise target."""
    best = None
    best_dist = float("inf")
    for cue in cues:
        dist = abs(cue["start"] - target)
        if dist < best_dist:
            best = cue["start"]
            best_dist = dist
    if best is not None and best_dist <= max_distance:
        return round(best, 3)
    return round(target, 3)


def build_clip_srt(cues: list[dict], clip_start: float, clip_duration: float, output_path: str) -> str:
    """Write a clip-local SRT file with timestamps shifted to start at zero.

    Cues that overlap the clip window are included and clamped to [0, clip_duration].
    Returns the output path.
    """
    clip_end = clip_start + clip_duration
    out_cues: list[srt.Subtitle] = []
    for cue in cues:
        if cue["end"] <= clip_start or cue["start"] >= clip_end:
            continue
        local_start = max(0.0, cue["start"] - clip_start)
        local_end = min(clip_duration, cue["end"] - clip_start)
        if local_end <= local_start:
            continue
        out_cues.append(
            srt.Subtitle(
                index=len(out_cues) + 1,
                start=timedelta(seconds=local_start),
                end=timedelta(seconds=local_end),
                content=cue["text"],
            )
        )

    if not out_cues:
        # Create a single placeholder cue so libass has something to render.
        out_cues.append(
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=clip_duration),
                content=" ",
            )
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(srt.compose(out_cues))
    return output_path
"""Transcript analysis, chunking, candidate validation, deduplication, and ranking."""

from __future__ import annotations

import math

from llm_providers import ProviderError, get_provider
from subtitle_utils import format_timestamp, nearest_cue_start


MIN_CLIP_SECONDS = 15.0
MAX_CLIP_SECONDS = 180.0
MAX_CHUNK_CHARS = 12_000
CHUNK_OVERLAP_CUES = 15
MIN_OVERLAP_SECONDS = 5.0
SNAP_DISTANCE_SECONDS = 2.0


class SelectionError(Exception):
    """Raised when clip selection cannot produce usable candidates."""


def build_transcript(cues: list[dict]) -> str:
    lines: list[str] = []
    for cue in cues:
        text = cue["text"].replace("\n", " ").strip()
        lines.append(
            f"[{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}] {text}"
        )
    return "\n".join(lines)


def chunk_cues(cues: list[dict], max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CUES) -> list[list[dict]]:
    """Split cues into chunks by character budget with a small overlap."""
    if not cues:
        return []

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0

    for cue in cues:
        line_len = len(cue["text"]) + 40
        if current and current_chars + line_len > max_chars:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            current_chars = sum(len(c["text"]) + 40 for c in current)
        current.append(cue)
        current_chars += line_len

    if current:
        chunks.append(current)

    return chunks


def _overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end_seconds"], b["end_seconds"]) - max(a["start_seconds"], b["start_seconds"]))


def dedupe_candidates(candidates: list[dict], min_overlap: float = MIN_OVERLAP_SECONDS) -> list[dict]:
    """Keep the highest-scoring candidates that do not substantially overlap."""
    ordered = sorted(candidates, key=lambda c: float(c.get("score", 0)), reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        if all(_overlap(candidate, existing) < min_overlap for existing in kept):
            kept.append(candidate)
    return kept


def _normalize_candidate(raw: dict, video_duration: float, cues: list[dict]) -> dict | None:
    """Validate and normalise a single LLM candidate. Returns None if unusable."""
    try:
        start = float(raw["start_seconds"])
        end = float(raw["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None

    if not math.isfinite(start) or not math.isfinite(end):
        return None

    # Snap start to the nearest cue boundary so clips do not begin mid-sentence.
    start = nearest_cue_start(cues, start, max_distance=SNAP_DISTANCE_SECONDS)

    if start < 0:
        start = 0.0

    # Clamp end to the video length.
    if video_duration and end > video_duration:
        end = video_duration

    if end <= start:
        return None

    duration = end - start
    if duration < MIN_CLIP_SECONDS or duration > MAX_CLIP_SECONDS:
        return None

    title = str(raw.get("title", "")).strip() or "Untitled moment"
    hook = str(raw.get("hook", "")).strip() or ""
    reason = str(raw.get("reason", "")).strip() or "Selected by the model."
    try:
        score = float(raw.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(10.0, score))

    return {
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "duration_seconds": round(duration, 2),
        "title": title[:200],
        "hook": hook[:300],
        "reason": reason[:800],
        "score": round(score, 2),
    }


def select_clips(
    cues: list[dict],
    video_duration: float,
    clip_count: int,
    provider_name: str,
) -> tuple[list[dict], str | None]:
    """Run the full selection pipeline.

    Returns (clips, warning). If fewer than clip_count strong clips are found,
    warning explains why.
    """
    if clip_count not in (3, 4):
        raise SelectionError("Clip count must be 3 or 4.")

    if not cues:
        raise SelectionError("No subtitle cues were parsed.")

    if video_duration < MIN_CLIP_SECONDS:
        raise SelectionError(
            f"The video is only {video_duration:.1f}s long. "
            f"At least {MIN_CLIP_SECONDS:.0f}s is required."
        )

    provider = get_provider(provider_name)
    chunks = chunk_cues(cues)
    all_raw: list[dict] = []
    errors: list[str] = []

    for idx, chunk in enumerate(chunks):
        transcript = build_transcript(chunk)
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        context = (
            f"This is chunk {idx + 1} of {len(chunks)}. "
            f"Its absolute time range is {format_timestamp(chunk_start)} to {format_timestamp(chunk_end)}. "
            f"The full video is {video_duration:.1f} seconds long. "
            f"Return up to {clip_count} strong candidates whose start times fall inside this chunk. "
            f"Pick each clip's natural length between {int(MIN_CLIP_SECONDS)} and {int(MAX_CLIP_SECONDS)} seconds."
        )
        try:
            raw_clips = provider.analyze(transcript, clip_count, context=context)
            all_raw.extend(raw_clips)
        except ProviderError as exc:
            errors.append(f"Chunk {idx + 1}: {exc}")

    if not all_raw:
        detail = "; ".join(errors) if errors else "The model returned no candidates."
        raise SelectionError(f"No clip candidates were produced. {detail}")

    normalized: list[dict] = []
    for raw in all_raw:
        candidate = _normalize_candidate(raw, video_duration, cues)
        if candidate is not None:
            normalized.append(candidate)

    if not normalized:
        raise SelectionError(
            "All model candidates were invalid (bad timestamps, wrong duration, or outside the video)."
        )

    deduped = dedupe_candidates(normalized)

    deduped.sort(key=lambda c: float(c["score"]), reverse=True)
    selected = deduped[:clip_count]
    selected.sort(key=lambda c: c["start_seconds"])

    warning = None
    if len(selected) < clip_count:
        warning = (
            f"The model found only {len(selected)} strong, non-overlapping clip(s) "
            f"out of the requested {clip_count}. Fewer clips are better than fabricated ones."
        )
        if errors:
            warning += " Some chunks also reported errors: " + "; ".join(errors)

    return selected, warning
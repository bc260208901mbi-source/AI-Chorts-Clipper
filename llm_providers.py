"""Groq and Gemini provider integrations with structured JSON output."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from pydantic import BaseModel, Field

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore


class ProviderError(Exception):
    """Raised when an LLM provider fails or returns unusable output."""


class ClipCandidateModel(BaseModel):
    start_seconds: float = Field(description="Absolute start time of the clip in seconds.")
    end_seconds: float = Field(description="Absolute end time of the clip in seconds. Must equal start_seconds + 60.")
    title: str = Field(description="Short, punchy title for the clip.")
    hook: str = Field(description="The opening hook that grabs attention.")
    reason: str = Field(description="Why this moment makes a strong standalone Short.")
    score: float = Field(description="Quality score from 0 to 10.")


class AnalysisResponse(BaseModel):
    clips: list[ClipCandidateModel]

SYSTEM_PROMPT = """You are an expert short-form video editor who creates viral YouTube Shorts by stitching together the best moments from a long video.

You will receive a timestamped transcript of a long video. Your job is to build Shorts — but NOT as one continuous cut. Each Short is assembled from multiple non-contiguous segments pulled from anywhere in the video (for example: 25s from the intro + 35s from the middle, or 15s + 20s + 25s from three different sections). You are the editor with the scissors: keep only what earns its place.

LENGTH PHILOSOPHY:
- 60 seconds is the target, not a cage. Aim for roughly 45–75 seconds.
- The idea wins over the clock. If the idea resolves at 38s, stop at 38s. If it needs 82s to land properly, take 82s.
- Never pad to hit a number, and never clip a payoff short to hit a number.
- Hard ceiling: 90 seconds. Hard floor: 30 seconds. If the best possible cut falls outside that, return fewer clips instead of forcing a weak one.

Prioritise:
- Interesting stories and complete narrative arcs.
- Surprising facts, revelations, or twists.
- Useful explanations and actionable advice.
- Strong hooks, engaging dialogue, and memorable conclusions.
- Segments that make sense without outside context.
- Pairs of moments that land harder together than apart (setup here, payoff there).

Avoid:
- Repetitive, boring, incomplete, or confusing passages.
- Segments that start in the middle of a sentence.
- Segments that end abruptly before the main point is delivered.
- Overlapping or nearly identical moments — both across clips and within a single clip.
- Cuts that make the speaker appear to say something they didn't (no Frankenstein sentences).

RULES FOR SEGMENTS:
1. Every segment must start AND end on a real subtitle cue boundary that exists in the transcript. Never invent timestamps.
2. Every segment must begin and end on a complete thought — never mid-sentence, never before the point lands.
3. Each clip must contain between 1 and 5 segments. Use the fewest cuts that still produce the strongest Short; prefer 2–3. A single continuous segment is allowed when one clean run is genuinely stronger than any assembly.
4. Within a clip, segments must not overlap.
5. Segments within a clip may come from any point in the video and may be reordered — put the strongest hook first, even if it originally appeared later.
6. Cuts must feel intentional, not accidental: each jump should be motivated by content (setup → payoff, claim → proof, question → answer). Do not cut away mid-argument.
7. The first segment must open with a hook. The final segment must deliver the conclusion or payoff. The clip is finished when the idea is finished — not when a timer hits zero.
8. `total_seconds` must equal the exact sum of all segment durations in that clip.

RULES FOR OUTPUT:
1. Return between 1 and {clip_count} clips. If there are not enough strong moments, return fewer.
2. Do not reuse the same source moment across different clips.
3. Do not invent timestamps. Only use timestamps that exist in the transcript.
4. Return ONLY valid JSON. No markdown, no commentary, no code fences.

JSON shape:
{{
  "clips": [
    {{
      "segments": [
        {{"start_seconds": 12.34, "end_seconds": 37.34}},
        {{"start_seconds": 75.10, "end_seconds": 110.10}}
      ],
      "total_seconds": 60.0,
      "title": "Short title",
      "hook": "Opening line of the first segment",
      "stitch": "One sentence on why these specific segments belong together and how the jump reads.",
      "reason": "Why this works as a standalone Short.",
      "score": 8.5
    }}
  ]
}}"""


def _clean_json_text(text: str) -> str:
    """Strip markdown fences and surrounding whitespace."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_response(raw: str) -> list[dict]:
    text = _clean_json_text(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Provider returned malformed JSON: {exc}") from exc

    try:
        parsed = AnalysisResponse.model_validate(data)
    except Exception as exc:
        raise ProviderError(f"Provider JSON did not match the expected schema: {exc}") from exc

    return [item.model_dump() for item in parsed.clips]


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        if Groq is None:
            raise ProviderError("The 'groq' package is not installed. Run: pip install groq")
        key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise ProviderError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=key, timeout=120.0)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    def analyze(self, transcript: str, clip_count: int, context: str = "") -> list[dict]:
        prompt = (
            SYSTEM_PROMPT.format(clip_count=clip_count)
            + "\n\n"
            + (context + "\n\n" if context else "")
            + "TRANSCRIPT:\n"
            + transcript
        )
        messages = [
            {"role": "system", "content": "You return only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return _parse_response(content)
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1.5)
                else:
                    break
        raise ProviderError(f"Groq request failed: {last_error}") from last_error


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        if genai is None:
            raise ProviderError("The 'google-genai' package is not installed. Run: pip install google-genai")
        key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    def analyze(self, transcript: str, clip_count: int, context: str = "") -> list[dict]:
        prompt = (
            SYSTEM_PROMPT.format(clip_count=clip_count)
            + "\n\n"
            + (context + "\n\n" if context else "")
            + "TRANSCRIPT:\n"
            + transcript
        )

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": AnalysisResponse.model_json_schema(),
                        "temperature": 0.4,
                    },
                )
                return _parse_response(response.text or "")
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1.5)
                else:
                    break
        raise ProviderError(f"Gemini request failed: {last_error}") from last_error


def get_provider(name: str) -> GroqProvider | GeminiProvider:
    normalized = (name or "").strip().lower()
    if normalized == "groq":
        return GroqProvider()
    if normalized == "gemini":
        return GeminiProvider()
    raise ProviderError(f"Unsupported provider: {name!r}. Choose 'groq' or 'gemini'.")
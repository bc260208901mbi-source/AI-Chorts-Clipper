"""Groq and Gemini provider integrations with structured JSON output."""

from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel, Field

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore


MIN_CLIP_SECONDS = 15.0
MAX_CLIP_SECONDS = 180.0


class ProviderError(Exception):
    """Raised when an LLM provider fails or returns unusable output."""


class ClipCandidateModel(BaseModel):
    start_seconds: float = Field(description="Absolute start time of the clip in seconds.")
    end_seconds: float = Field(
        description=(
            "Absolute end time of the clip in seconds. Must be greater than start_seconds. "
            f"Clip length must be between {int(MIN_CLIP_SECONDS)} and {int(MAX_CLIP_SECONDS)} seconds."
        )
    )
    title: str = Field(description="Short, punchy title for the clip.")
    hook: str = Field(description="The opening hook that grabs attention.")
    reason: str = Field(description="Why this moment makes a strong standalone Short.")
    score: float = Field(description="Quality score from 0 to 10.")


class AnalysisResponse(BaseModel):
    clips: list[ClipCandidateModel]


SYSTEM_PROMPT = """You are an expert short-form video editor who creates viral YouTube Shorts.

You will receive a timestamped transcript of a long video. Identify the strongest standalone moments that can each become a Short.

Prioritise:
- Interesting stories and complete narrative arcs.
- Surprising facts, revelations, or twists.
- Useful explanations and actionable advice.
- Strong hooks, engaging dialogue, and memorable conclusions.
- Segments that make sense without outside context.

Avoid:
- Repetitive, boring, incomplete, or confusing passages.
- Clips that start in the middle of a sentence.
- Clips that end abruptly before the main point is delivered.
- Overlapping or nearly identical moments.

DURATION RULES (important):
1. Choose the natural duration of each Short. Do NOT pad or truncate to a fixed length.
2. Minimum total duration: {min_s}s. Maximum total duration: {max_s}s.
3. Total duration is the SUM of all selected segments, not the time between the first start and final end.
4. end_seconds MUST be greater than start_seconds for every segment.
5. Start times should align with subtitle cue boundaries and complete thoughts.
6. Do NOT invent timestamps. Only use timestamps that exist in the transcript.
7. Return between 1 and {clip_count} Shorts. If there are not enough strong moments, return fewer.

MULTI-SEGMENT CLIPS:
1. Each Short may contain 1–4 separate segments from the original video.
2. A segment has its own start_seconds and end_seconds.
3. Combine useful moments from different parts of the video into ONE Short when doing so improves the story.
4. Skip irrelevant, repetitive, silent, or boring material between selected segments.
5. Arrange segments in chronological order.
6. Ensure the combined segments form a coherent story with a strong opening and satisfying conclusion.
7. Prefer a continuous segment when it communicates the idea effectively. Do not create unnecessary cuts.
8. The total duration of all segments combined must satisfy the minimum and maximum duration rules.
9. Do not select overlapping segments or repeat the same content.
10. Each Short's segments must come from the same original video.
11. The first segment should usually contain the hook. Later segments should develop the idea, provide evidence, or deliver the conclusion.
12. Choose natural cut points that avoid cutting off words or important dialogue.
13. Do not assume gaps between segments will be included in the final Short.
14. The video-processing program will concatenate the selected segments in the order returned.

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.
- Do not include Markdown, explanations, or commentary outside the JSON.
- Use numeric values for timestamps and scores.
- Every Short must contain a "segments" array, even if it has only one segment.
- The score should represent the estimated quality of the complete assembled Short, from 1 to 10.

JSON shape:
{{
  "clips": [
    {{
      "title": "Short title",
      "hook": "Opening line or hook",
      "reason": "Why this assembled Short works as a standalone video",
      "score": 8.5,
      "segments": [
        {{
          "start_seconds": 0.0,
          "end_seconds": 10.0
        }},
        {{
          "start_seconds": 30.0,
          "end_seconds": 45.0
        }},
        {{
          "start_seconds": 75.0,
          "end_seconds": 90.0
        }}
      ]
    }}
  ]
}}"""


def _build_prompt(clip_count: int, context: str, transcript: str) -> str:
    return (
        SYSTEM_PROMPT.format(
            clip_count=clip_count,
            min_s=int(MIN_CLIP_SECONDS),
            max_s=int(MAX_CLIP_SECONDS),
        )
        + "\n\n"
        + (context + "\n\n" if context else "")
        + "TRANSCRIPT:\n"
        + transcript
    )


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
        prompt = _build_prompt(clip_count, context, transcript)
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
        prompt = _build_prompt(clip_count, context, transcript)

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
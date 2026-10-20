"""Flask entry point for AI Shorts Clipper."""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from clip_selector import SelectionError, select_clips
from llm_providers import ProviderError
from subtitle_utils import SubtitleError, build_clip_srt, format_timestamp_short, parse_srt_file
from video_processor import VideoError, check_ffmpeg, probe_duration, render_clip

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
ALLOWED_SRT = {".srt"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job() -> dict:
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "dir": str(job_dir),
        "status": "queued",
        "progress": 0,
        "message": "Queued.",
        "clips": [],
        "error": None,
        "warning": None,
        "created_at": time.time(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _cleanup_old_jobs(max_age_seconds: int = 3 * 3600) -> None:
    cutoff = time.time() - max_age_seconds
    with _jobs_lock:
        stale = [jid for jid, job in _jobs.items() if job["created_at"] < cutoff]
        for jid in stale:
            _jobs.pop(jid, None)
    for jid in stale:
        shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)


def _allowed(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


def _process_job(job_id: str, video_path: str, srt_path: str, provider: str, clip_count: int) -> None:
    try:
        _update_job(job_id, status="analyzing", progress=10, message="Parsing subtitles…")
        cues = parse_srt_file(srt_path)

        _update_job(job_id, progress=20, message="Reading video duration…")
        duration = probe_duration(video_path)

        _update_job(
            job_id,
            status="selecting",
            progress=30,
            message="Asking the AI to find strong moments…",
        )
        clips, warning = select_clips(cues, duration, clip_count, provider)

        _update_job(
            job_id,
            status="rendering",
            progress=45,
            message=f"Rendering {len(clips)} clip(s) with FFmpeg…",
            warning=warning,
        )

        results: list[dict] = []
        total = max(1, len(clips))
        for idx, clip in enumerate(clips):
            start = clip["start_seconds"]
            end = clip["end_seconds"]
            clip_duration = round(end - start, 3)
            output_name = f"short_{idx + 1:02d}.mp4"
            output_path = os.path.join(_jobs[job_id]["dir"], output_name)
            clip_srt_path = os.path.join(_jobs[job_id]["dir"], f"short_{idx + 1:02d}.srt")

            build_clip_srt(cues, start, clip_duration, clip_srt_path)

            _update_job(
                job_id,
                progress=45 + int(45 * idx / total),
                message=f"Rendering clip {idx + 1} of {len(clips)} ({clip_duration:.0f}s)…",
            )

            render_clip(video_path, clip_srt_path, start, output_path, duration=clip_duration)

            results.append(
                {
                    **clip,
                    "duration_seconds": clip_duration,
                    "duration_display": f"{int(round(clip_duration))}s",
                    "start_display": format_timestamp_short(start),
                    "end_display": format_timestamp_short(end),
                    "filename": output_name,
                    "url": f"/media/{job_id}/{output_name}",
                    "download_url": f"/download/{job_id}/{output_name}",
                }
            )

        _update_job(
            job_id,
            status="done",
            progress=100,
            message=f"Done. {len(results)} clip(s) rendered.",
            clips=results,
        )

    except (SubtitleError, SelectionError, ProviderError, VideoError) as exc:
        _update_job(job_id, status="error", error=str(exc), message=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        _update_job(job_id, status="error", error=f"Unexpected error: {exc}", message="Unexpected error.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        check_ffmpeg()
        ffmpeg_ok = True
        ffmpeg_error = None
    except VideoError as exc:
        ffmpeg_ok = False
        ffmpeg_error = str(exc)

    return jsonify(
        {
            "ffmpeg": ffmpeg_ok,
            "ffmpeg_error": ffmpeg_error,
            "groq_key": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "gemini_key": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        }
    )


@app.route("/api/generate", methods=["POST"])
def generate():
    _cleanup_old_jobs()

    if "video" not in request.files or "srt" not in request.files:
        return jsonify({"error": "Both a video file and an SRT file are required."}), 400

    video_file = request.files["video"]
    srt_file = request.files["srt"]
    provider = (request.form.get("provider") or "").strip().lower()

    try:
        clip_count = int(request.form.get("clip_count", "3"))
    except ValueError:
        return jsonify({"error": "Clip count must be 3 or 4."}), 400

    if clip_count not in (3, 4):
        return jsonify({"error": "Clip count must be 3 or 4."}), 400

    if provider not in ("groq", "gemini"):
        return jsonify({"error": "Choose either Groq or Gemini."}), 400

    if not video_file.filename or not srt_file.filename:
        return jsonify({"error": "Both files must have valid filenames."}), 400

    if not _allowed(video_file.filename, ALLOWED_VIDEO):
        return jsonify({"error": f"Unsupported video type. Allowed: {', '.join(sorted(ALLOWED_VIDEO))}"}), 400

    if not _allowed(srt_file.filename, ALLOWED_SRT):
        return jsonify({"error": "The subtitle file must be an .srt file."}), 400

    try:
        check_ffmpeg()
    except VideoError as exc:
        return jsonify({"error": str(exc)}), 500

    job = _new_job()
    job_dir = job["dir"]

    safe_video_name = secure_filename(video_file.filename) or "input.mp4"
    safe_srt_name = secure_filename(srt_file.filename) or "input.srt"
    video_path = os.path.join(job_dir, safe_video_name)
    srt_path = os.path.join(job_dir, safe_srt_name)

    try:
        video_file.save(video_path)
        srt_file.save(srt_path)
    except OSError as exc:
        _update_job(job["id"], status="error", error=f"Could not save uploads: {exc}")
        return jsonify({"error": "Could not save uploaded files."}), 500

    if os.path.getsize(video_path) == 0 or os.path.getsize(srt_path) == 0:
        _update_job(job["id"], status="error", error="One of the uploaded files is empty.")
        return jsonify({"error": "Uploaded files must not be empty."}), 400

    thread = threading.Thread(
        target=_process_job,
        args=(job["id"], video_path, srt_path, provider, clip_count),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job["id"]}), 202


@app.route("/api/job/<job_id>")
def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(
        {
            "id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "clips": job["clips"],
            "error": job["error"],
            "warning": job["warning"],
        }
    )


@app.route("/media/<job_id>/<filename>")
def media(job_id: str, filename: str):
    job = _get_job(job_id)
    if not job:
        abort(404)
    directory = job["dir"]
    return send_from_directory(directory, filename, conditional=True)


@app.route("/download/<job_id>/<filename>")
def download(job_id: str, filename: str):
    job = _get_job(job_id)
    if not job:
        abort(404)
    directory = job["dir"]
    return send_from_directory(directory, filename, as_attachment=True, conditional=True)


if __name__ == "__main__":
    try:
        check_ffmpeg()
    except VideoError as exc:
        print(f"[WARN] {exc}")

    if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("[WARN] No API key found. Copy .env.example to .env and add a key.")

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
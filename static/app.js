const form = document.getElementById("generate-form");
const submitBtn = document.getElementById("submit-btn");
const statusBox = document.getElementById("status-box");
const statusMessage = document.getElementById("status-message");
const statusPercent = document.getElementById("status-percent");
const progressBar = document.getElementById("progress-bar");
const errorBox = document.getElementById("error-box");
const resultsSection = document.getElementById("results");
const resultsList = document.getElementById("results-list");
const warningBox = document.getElementById("warning-box");

let pollTimer = null;

function setBusy(busy) {
  submitBtn.disabled = busy;
  submitBtn.textContent = busy ? "Working…" : "Analyze & Generate Shorts";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function updateProgress(percent, message) {
  const p = Math.max(0, Math.min(100, Number(percent) || 0));
  progressBar.style.width = `${p}%`;
  statusPercent.textContent = `${p}%`;
  statusMessage.textContent = message || "Working…";
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function renderClips(clips, warning) {
  resultsList.innerHTML = "";

  if (warning) {
    warningBox.textContent = warning;
    warningBox.classList.remove("hidden");
  } else {
    warningBox.textContent = "";
    warningBox.classList.add("hidden");
  }

  if (!clips || clips.length === 0) {
    resultsSection.classList.remove("hidden");
    resultsList.innerHTML = '<p class="clip-reason">No clips were generated.</p>';
    return;
  }

  clips.forEach((clip) => {
    const card = document.createElement("article");
    card.className = "clip-card";

    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = clip.url;
    video.setAttribute("playsinline", "");
    card.appendChild(video);

    const body = document.createElement("div");
    body.className = "clip-body";

    const title = document.createElement("h3");
    title.className = "clip-title";
    title.textContent = clip.title || "Untitled clip";
    body.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "clip-meta";
        const durationLabel =
      clip.duration_display ||
      `${Math.round((clip.end_seconds || 0) - (clip.start_seconds || 0))}s`;
    const pills = [
      `${clip.start_display} → ${clip.end_display}`,
      durationLabel,
      `Score ${Number(clip.score || 0).toFixed(1)}`,
    ];
    pills.forEach((text) => {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = text;
      meta.appendChild(pill);
    });
    body.appendChild(meta);

    if (clip.hook) {
      const hook = document.createElement("p");
      hook.className = "clip-hook";
      hook.textContent = clip.hook;
      body.appendChild(hook);
    }

    const reason = document.createElement("p");
    reason.className = "clip-reason";
    reason.textContent = clip.reason || "";
    body.appendChild(reason);

    const download = document.createElement("a");
    download.className = "download";
    download.href = clip.download_url;
    download.textContent = "Download MP4";
    body.appendChild(download);

    card.appendChild(body);
    resultsList.appendChild(card);
  });

  resultsSection.classList.remove("hidden");
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/job/${jobId}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Status check failed (${response.status}).`);
    }
    const job = await response.json();

    updateProgress(job.progress, job.message);

    if (job.status === "done") {
      stopPolling();
      setBusy(false);
      updateProgress(100, job.message || "Done.");
      renderClips(job.clips, job.warning);
      statusBox.classList.add("hidden");
    } else if (job.status === "error") {
      stopPolling();
      setBusy(false);
      showError(job.error || "Something went wrong.");
      statusBox.classList.add("hidden");
    }
  } catch (err) {
    stopPolling();
    setBusy(false);
    showError(err.message || "Could not check job status.");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  stopPolling();

  const video = document.getElementById("video").files[0];
  const srt = document.getElementById("srt").files[0];

  if (!video || !srt) {
    showError("Please choose both a video file and an SRT file.");
    return;
  }

  const formData = new FormData();
  formData.append("video", video);
  formData.append("srt", srt);
  formData.append("provider", document.getElementById("provider").value);
  formData.append("clip_count", document.getElementById("clip_count").value);

  setBusy(true);
  resultsSection.classList.add("hidden");
  statusBox.classList.remove("hidden");
  updateProgress(5, "Uploading…");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.error || `Upload failed (${response.status}).`);
    }

    updateProgress(8, "Upload received. Starting analysis…");
    pollTimer = setInterval(() => pollJob(data.job_id), 1500);
    pollJob(data.job_id);
  } catch (err) {
    setBusy(false);
    statusBox.classList.add("hidden");
    showError(err.message || "Could not start the job.");
  }
});
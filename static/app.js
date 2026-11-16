const $ = id => document.getElementById(id);
const input = $("image-input");
const dropzone = $("dropzone");
const form = $("upload-form");
const button = $("analyze-btn");
const buttonLabel = button.querySelector("span");
let previewUrl = null;

$("today").textContent = new Intl.DateTimeFormat(undefined, {
  month: "short", day: "numeric", year: "numeric"
}).format(new Date());

function safe(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}

function formatDate(value) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "—" :
    parsed.toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function showError(message = "") {
  $("error-message").textContent = message;
}

function setFile(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    showError("Please choose a PNG or JPG image.");
    input.value = "";
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("This image is larger than 10 MB. Choose a smaller file.");
    input.value = "";
    return;
  }

  showError("");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("file-preview").innerHTML =
    '<span class="selected-file"><img src="' + previewUrl +
    '" alt="Selected X-ray preview"><span><b>' + safe(file.name) +
    '</b><small>' + (file.size / (1024 * 1024)).toFixed(2) +
    ' MB · Ready to analyze</small></span><span class="file-check" aria-hidden="true">✓</span></span>';
  $("empty-result").classList.remove("hidden");
  $("result-content").classList.add("hidden");
}

input.addEventListener("change", () => setFile(input.files[0]));

["dragenter", "dragover"].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  setFile(file);
});

async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || "The service could not complete the request.");
  return payload;
}

async function refresh() {
  const status = document.querySelector(".system-status");
  try {
    const [healthResponse, statsResponse, historyResponse] = await Promise.all([
      fetch("/api/health"), fetch("/api/stats"), fetch("/api/history")
    ]);
    const [health, stats, rows] = await Promise.all([
      readJson(healthResponse), readJson(statsResponse), readJson(historyResponse)
    ]);

    $("model-status").textContent = health.model_ready ? "Model ready" : "Model not trained";
    status.classList.toggle("ready", Boolean(health.model_ready));
    $("total-count").textContent = stats.total ?? "—";
    $("normal-count").textContent = stats.normal ?? "—";
    $("pneumonia-count").textContent = stats.pneumonia ?? "—";

    $("history-body").innerHTML = Array.isArray(rows) && rows.length
      ? rows.slice(0, 6).map(row => {
          const prediction = String(row.prediction || "Unknown").toLowerCase();
          const tone = prediction === "normal" ? "normal" : prediction === "pneumonia" ? "pneumonia" : "";
          const confidence = Number(row.confidence);
          return '<tr><td class="file-cell">' + safe(row.filename || "Image") +
            '</td><td><span class="pill ' + tone + '">' + safe(row.prediction || "—") +
            '</span></td><td>' + (Number.isFinite(confidence) ? confidence.toFixed(1) + "%" : "—") +
            '</td><td>' + formatDate(row.created_at) +
            '</td><td><span class="table-status"><i></i> Completed</span></td></tr>';
        }).join("")
      : '<tr><td colspan="5" class="table-empty">No analyses yet. Your activity will appear here.</td></tr>';
  } catch (error) {
    $("model-status").textContent = "Service unavailable";
    status.classList.remove("ready");
    $("history-body").innerHTML =
      '<tr><td colspan="5" class="table-empty">Workspace data is temporarily unavailable. Please refresh in a moment.</td></tr>';
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) {
    showError("Choose a chest X-ray image before starting the analysis.");
    input.focus();
    return;
  }

  const data = new FormData();
  data.append("image", file);
  button.disabled = true;
  button.classList.add("is-loading");
  buttonLabel.textContent = "Analyzing image…";
  showError("");

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: data });
    const result = await readJson(response);
    $("empty-result").classList.add("hidden");
    $("result-content").classList.remove("hidden");
    $("prediction").textContent = result.prediction || "No prediction";
    $("prediction").classList.toggle("alert", result.prediction === "PNEUMONIA");
    const confidence = Math.max(0, Math.min(100, Number(result.confidence) || 0));
    $("confidence").textContent = confidence.toFixed(1) + "%";
    $("confidence-bar").style.width = confidence + "%";
    $("result-note").textContent =
      "Model output for " + (result.filename || file.name) +
      ". This score is a model confidence value, not a patient's probability of disease.";
    await refresh();
  } catch (error) {
    showError(error.message || "Analysis failed. Please try again.");
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
    buttonLabel.textContent = "Run AI analysis";
  }
});

refresh();

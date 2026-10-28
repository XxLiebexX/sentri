const API_BASE = "http://localhost:8000";

const modeTabs = document.querySelectorAll(".mode-tab");
const snippetPanel = document.getElementById("snippet-input-panel");
const repoPanel = document.getElementById("repo-input-panel");

const resultsContent = document.getElementById("results-content");
const scanResultsContent = document.getElementById("scan-results-content");
const emptyState = document.getElementById("empty-state");

modeTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    modeTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const mode = tab.dataset.mode;
    snippetPanel.hidden = mode !== "snippet";
    repoPanel.hidden = mode !== "repo";
    resultsContent.hidden = true;
    scanResultsContent.hidden = true;
    emptyState.hidden = false;
  });
});

// ---- Snippet review (existing) ----

const codeInput = document.getElementById("code-input");
const submitBtn = document.getElementById("submit-btn");
const statusMsg = document.getElementById("status-msg");
const finalBanner = document.getElementById("final-banner");
const finalSourceBadge = document.getElementById("final-source-badge");
const finalReason = document.getElementById("final-reason");

submitBtn.addEventListener("click", runReview);

async function runReview() {
  const code = codeInput.value.trim();
  if (!code) {
    setStatus(statusMsg, "Paste some code first.", true);
    return;
  }

  setStatus(statusMsg, "Reviewing...", false);
  submitBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderResults(data);
    setStatus(statusMsg, "", false);
  } catch (e) {
    setStatus(statusMsg, e.message, true);
  } finally {
    submitBtn.disabled = false;
  }
}

function setStatus(el, msg, isError) {
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

function renderResults(data) {
  emptyState.hidden = true;
  scanResultsContent.hidden = true;
  resultsContent.hidden = false;

  finalBanner.classList.remove("local", "gemini");
  finalBanner.classList.add(data.final_source);
  finalSourceBadge.textContent = data.final_source === "gemini" ? "Gemini" : "Local model";
  finalReason.textContent = data.escalated
    ? `Escalated - local confidence was below ${Math.round(data.confidence_threshold * 100)}%, or it flagged an issue`
    : "Local model was confident and found nothing - no escalation needed";

  setVerdictBadge("local-verdict-badge", data.local.verdict);
  document.getElementById("local-cwe").textContent = data.local.cwe || "-";
  document.getElementById("local-explanation").textContent = data.local.explanation || "-";

  const confidencePct = Math.round((data.local.confidence ?? 0) * 100);
  document.getElementById("local-confidence-fill").style.width = `${confidencePct}%`;
  document.getElementById("local-confidence-text").textContent = `${confidencePct}%`;

  setVerdictBadge("gemini-verdict-badge", data.gemini.verdict);
  document.getElementById("gemini-cwe").textContent = data.gemini.cwe || "-";
  document.getElementById("gemini-explanation").textContent = data.gemini.explanation || "-";

  document.getElementById("latency-note").textContent = `Reviewed in ${data.latency_ms}ms`;
}

function setVerdictBadge(elId, verdict) {
  const el = document.getElementById(elId);
  el.textContent = verdict;
  el.classList.remove("safe", "vulnerable");
  if (verdict === "safe") el.classList.add("safe");
  if (verdict === "vulnerable") el.classList.add("vulnerable");
}

// ---- Repo scan (new) ----

const repoUrlInput = document.getElementById("repo-url-input");
const maxFilesInput = document.getElementById("max-files-input");
const scanBtn = document.getElementById("scan-btn");
const scanStatusMsg = document.getElementById("scan-status-msg");

scanBtn.addEventListener("click", runScan);

async function runScan() {
  const repoUrl = repoUrlInput.value.trim();
  if (!repoUrl) {
    setStatus(scanStatusMsg, "Paste a repo URL first.", true);
    return;
  }

  const maxFiles = parseInt(maxFilesInput.value, 10) || 5;

  setStatus(scanStatusMsg, "Scanning... this can take a minute or two (cloning, project overview, per-file review).", false);
  scanBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/scan-repo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_url: repoUrl, max_files: maxFiles }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderScanResults(data);
    setStatus(scanStatusMsg, "", false);
  } catch (e) {
    setStatus(scanStatusMsg, e.message, true);
  } finally {
    scanBtn.disabled = false;
  }
}

function renderScanResults(data) {
  emptyState.hidden = true;
  resultsContent.hidden = true;
  scanResultsContent.hidden = false;

  document.getElementById("scan-project-type").textContent = data.project_type || "Unknown project type";
  document.getElementById("scan-purpose").textContent = data.purpose || "";
  document.getElementById("scan-meta").textContent =
    `${data.files_scanned} files scanned (${data.files_skipped} skipped) - ${data.findings.length} findings`;

  const listEl = document.getElementById("scan-findings-list");
  listEl.innerHTML = "";

  if (data.findings.length === 0) {
    listEl.innerHTML = "<p class=\'no-findings\'>No issues found in the scanned files.</p>";
    return;
  }

  const severityOrder = { error: 0, warning: 1, info: 2 };
  const sorted = [...data.findings].sort((a, b) => (severityOrder[a.severity] ?? 3) - (severityOrder[b.severity] ?? 3));

  for (const finding of sorted) {
    const item = document.createElement("div");
    item.className = `finding-item severity-${finding.severity}`;
    item.innerHTML = `
      <div class="finding-header">
        <span class="finding-file">${finding.file}:${finding.line}</span>
        <span class="finding-source">${finding.source}</span>
      </div>
      <div class="finding-category">${finding.category}</div>
      <div class="finding-message">${finding.message}</div>
    `;
    listEl.appendChild(item);
  }
}

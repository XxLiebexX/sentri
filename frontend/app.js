const API_BASE = "http://localhost:8000";

const codeInput = document.getElementById("code-input");
const submitBtn = document.getElementById("submit-btn");
const statusMsg = document.getElementById("status-msg");

const emptyState = document.getElementById("empty-state");
const resultsContent = document.getElementById("results-content");

const finalBanner = document.getElementById("final-banner");
const finalSourceBadge = document.getElementById("final-source-badge");
const finalReason = document.getElementById("final-reason");

submitBtn.addEventListener("click", runReview);

async function runReview() {
  const code = codeInput.value.trim();
  if (!code) {
    setStatus("Paste some code first.", true);
    return;
  }

  setStatus("Reviewing...", false);
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
    setStatus("", false);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    submitBtn.disabled = false;
  }
}

function setStatus(msg, isError) {
  statusMsg.textContent = msg;
  statusMsg.classList.toggle("error", isError);
}

function renderResults(data) {
  emptyState.hidden = true;
  resultsContent.hidden = false;

  // Final banner
  finalBanner.classList.remove("local", "gemini");
  finalBanner.classList.add(data.final_source);
  finalSourceBadge.textContent = data.final_source === "gemini" ? "Gemini" : "Local model";
  finalReason.textContent = data.escalated
    ? `Escalated — local confidence was below ${Math.round(data.confidence_threshold * 100)}%, or it flagged an issue`
    : "Local model was confident and found nothing — no escalation needed";

  // Local card
  setVerdictBadge("local-verdict-badge", data.local.verdict);
  document.getElementById("local-cwe").textContent = data.local.cwe || "—";
  document.getElementById("local-explanation").textContent = data.local.explanation || "—";

  const confidencePct = Math.round((data.local.confidence ?? 0) * 100);
  document.getElementById("local-confidence-fill").style.width = `${confidencePct}%`;
  document.getElementById("local-confidence-text").textContent = `${confidencePct}%`;

  // Gemini card
  setVerdictBadge("gemini-verdict-badge", data.gemini.verdict);
  document.getElementById("gemini-cwe").textContent = data.gemini.cwe || "—";
  document.getElementById("gemini-explanation").textContent = data.gemini.explanation || "—";

  document.getElementById("latency-note").textContent =
    `Reviewed in ${data.latency_ms}ms`;
}

function setVerdictBadge(elId, verdict) {
  const el = document.getElementById(elId);
  el.textContent = verdict;
  el.classList.remove("safe", "vulnerable");
  if (verdict === "safe") el.classList.add("safe");
  if (verdict === "vulnerable") el.classList.add("vulnerable");
}

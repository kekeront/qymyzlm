"use strict";

// ---------- Tabs ----------
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");
tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    tabPanels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---------- Model info ----------
async function loadModelInfo() {
  const statusEl = document.getElementById("model-status");
  try {
    const res = await fetch("/api/model");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const gen = data.generative || {};
    const emb = data.embedding || {};
    statusEl.textContent = "";
    statusEl.appendChild(
      buildStatusSpan("generative", gen.base_model, gen.loaded, gen.device)
    );
    statusEl.appendChild(
      buildStatusSpan("embedding", emb.model, emb.loaded, null)
    );
  } catch (err) {
    statusEl.textContent = `could not reach /api/model (${err.message})`;
  }
}
loadModelInfo();

function buildStatusSpan(label, modelName, loaded, device) {
  const span = document.createElement("span");
  const code = document.createElement("code");
  code.textContent = modelName ?? "?";
  span.appendChild(document.createTextNode(`${label}: `));
  span.appendChild(code);
  const loadedText = loaded ? "loaded" : "not loaded";
  const suffix = device ? `, device: ${device}` : "";
  span.appendChild(document.createTextNode(` (${loadedText}${suffix})`));
  return span;
}

// ---------- Prompt chips (generate) ----------
document.getElementById("prompt-chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  document.getElementById("prompt").value = chip.textContent;
});

// ---------- Generate ----------
const generateBtn = document.getElementById("generate-btn");
const stopBtn = document.getElementById("stop-btn");
const outputEl = document.getElementById("generate-output");
const statsEl = document.getElementById("generate-stats");
let abortController = null;

generateBtn.addEventListener("click", runGenerate);

// Falls back to `defaultValue` when the field is blank or not a valid number, so an
// empty temperature/top_p input doesn't silently become 0 (accidental greedy decoding).
function numberOrDefault(elementId, defaultValue) {
  const raw = document.getElementById(elementId).value;
  if (raw.trim() === "") return defaultValue;
  const n = Number(raw);
  return Number.isNaN(n) ? defaultValue : n;
}

async function runGenerate() {
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) {
    outputEl.textContent = "(enter a prompt first)";
    return;
  }

  const body = {
    prompt,
    max_new_tokens: Number(document.getElementById("max-tokens").value) || 128,
    temperature: numberOrDefault("temperature", 0.8),
    top_p: numberOrDefault("top-p", 0.95),
    engram: document.getElementById("engram-toggle").checked,
    chat: document.getElementById("chat-toggle").checked,
  };

  outputEl.textContent = "";
  statsEl.textContent = "";
  generateBtn.disabled = true;
  stopBtn.disabled = false;

  abortController = new AbortController();

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: abortController.signal,
    });

    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let msg;
        try {
          msg = JSON.parse(payload);
        } catch {
          continue;
        }
        handleGenerateEvent(msg);
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      statsEl.textContent = "(stopped by user)";
    } else {
      outputEl.textContent += `\n[error: ${err.message}]`;
    }
  } finally {
    generateBtn.disabled = false;
    stopBtn.disabled = true;
    abortController = null;
    // Header status (loaded/engram_grafted) can change after a run (lazy load,
    // lazy graft) — refresh it instead of leaving the load-time snapshot stale.
    loadModelInfo();
  }
}

function handleGenerateEvent(msg) {
  if (msg.error) {
    outputEl.textContent += `\n[error: ${msg.error}]`;
    return;
  }
  if (msg.done) {
    const s = msg.stats || {};
    statsEl.textContent =
      `tokens: ${s.n_tokens ?? "?"} · elapsed: ${fmtNum(s.elapsed_s)}s · ` +
      `${fmtNum(s.tok_per_s)} tok/s · engram: ${s.engram ?? "?"} · model: ${s.model ?? "?"}`;
    return;
  }
  if (typeof msg.token === "string") {
    outputEl.textContent += msg.token;
  }
}

function fmtNum(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "?";
  return n.toFixed(2);
}

stopBtn.addEventListener("click", () => {
  if (abortController) abortController.abort();
});

// ---------- Embed ----------
const embedRowsEl = document.getElementById("embed-rows");
const embedAddRowBtn = document.getElementById("embed-add-row");
const embedBtn = document.getElementById("embed-btn");
const embedMetaEl = document.getElementById("embed-meta");
const embedResultEl = document.getElementById("embed-result");

function addEmbedRow(text = "") {
  const row = document.createElement("div");
  row.className = "embed-row";

  const input = document.createElement("input");
  input.type = "text";
  input.value = text;
  input.placeholder = "Text to embed…";

  const removeBtn = document.createElement("button");
  removeBtn.textContent = "×";
  removeBtn.className = "row-remove";
  removeBtn.title = "Remove this row";
  removeBtn.addEventListener("click", () => {
    row.remove();
  });

  row.appendChild(input);
  row.appendChild(removeBtn);
  embedRowsEl.appendChild(row);
  return input;
}

// seed with two empty rows
addEmbedRow();
addEmbedRow();

embedAddRowBtn.addEventListener("click", () => addEmbedRow());

document.getElementById("embed-chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  addEmbedRow(chip.textContent);
});

embedBtn.addEventListener("click", runEmbed);

async function runEmbed() {
  const texts = Array.from(embedRowsEl.querySelectorAll("input"))
    .map((el) => el.value.trim())
    .filter((t) => t.length > 0);

  if (texts.length === 0) {
    embedMetaEl.textContent = "(add at least one non-empty text)";
    embedResultEl.innerHTML = "";
    return;
  }

  const mode = document.querySelector('input[name="embed-mode"]:checked').value;

  embedBtn.disabled = true;
  embedMetaEl.textContent = "embedding…";
  embedResultEl.textContent = "";

  try {
    const res = await fetch("/api/embed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts, mode }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    embedMetaEl.textContent = `model: ${data.model} · dim: ${data.dim} · texts: ${texts.length}`;
    renderSimilarityMatrix(texts, data.similarity);
  } catch (err) {
    embedMetaEl.textContent = `[error: ${err.message}]`;
  } finally {
    embedBtn.disabled = false;
  }
}

function renderSimilarityMatrix(labels, matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    embedResultEl.textContent = "";
    return;
  }

  const table = document.createElement("table");
  table.className = "sim-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th"));
  labels.forEach((label, i) => {
    const th = document.createElement("th");
    th.textContent = `#${i + 1}`;
    th.title = label;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  matrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    const rowHeader = document.createElement("th");
    rowHeader.textContent = `#${i + 1}`;
    rowHeader.title = labels[i] ?? "";
    tr.appendChild(rowHeader);

    row.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value.toFixed(2);
      td.style.backgroundColor = simColor(value);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  const legend = document.createElement("div");
  legend.className = "sim-legend";
  labels.forEach((label, i) => {
    const item = document.createElement("div");
    item.className = "sim-legend-item";
    item.textContent = `#${i + 1}: ${label}`;
    legend.appendChild(item);
  });

  embedResultEl.textContent = "";
  embedResultEl.appendChild(table);
  embedResultEl.appendChild(legend);
}

function simColor(value) {
  // value in roughly [-1, 1]; clamp to [0, 1] for a cosine-similarity heat scale
  const t = Math.max(0, Math.min(1, (value + 1) / 2));
  const hue = 220 - t * 220; // blue (low) -> red (high)
  const light = 88 - t * 38;
  return `hsl(${hue}, 70%, ${light}%)`;
}

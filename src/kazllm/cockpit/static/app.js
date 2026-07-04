// Campaign Mission Control — vanilla JS, no frameworks, no CDN.
// Consumes GET /api/status (same-origin) and renders it defensively:
// every field may be null/missing and must never throw.

(function () {
  "use strict";

  var POLL_MS = 15000;
  var STALE_AFTER_MS = POLL_MS * 3; // conn dot goes amber if no fresh data this long
  var lastFetchOkAt = null;
  var lastPayload = null;

  var els = {
    updatedAgo: document.getElementById("updated-ago"),
    connDot: document.getElementById("conn-dot"),
    errorBanner: document.getElementById("error-banner"),
    errorList: document.getElementById("error-list"),

    focusPrimary: document.getElementById("focus-primary"),
    focusStatus: document.getElementById("focus-status"),
    focusParkedSummary: document.getElementById("focus-parked-summary"),
    focusParkedList: document.getElementById("focus-parked-list"),

    computeKernel: document.getElementById("compute-kernel"),
    computeVersion: document.getElementById("compute-version"),
    computeStatus: document.getElementById("compute-status"),
    computeFp16: document.getElementById("compute-fp16"),
    computeRunningFor: document.getElementById("compute-running-for"),
    computeNote: document.getElementById("compute-note"),

    claimsChips: document.getElementById("claims-chips"),
    claimsLive: document.getElementById("claims-live"),
    claimsRetracted: document.getElementById("claims-retracted"),

    kbTiles: document.getElementById("kb-tiles"),

    ladderList: document.getElementById("ladder-list"),

    reposTbody: document.getElementById("repos-tbody"),
  };

  function text(node, value) {
    node.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined && txt !== null) n.textContent = txt;
    return n;
  }

  // ---------- time formatting ----------

  function fmtAgo(ms) {
    if (ms < 0) ms = 0;
    var s = Math.floor(ms / 1000);
    if (s < 5) return "just now";
    if (s < 60) return s + "s ago";
    var m = Math.floor(s / 60);
    if (m < 60) return m + "m ago";
    var h = Math.floor(m / 60);
    return h + "h ago";
  }

  function parseIso(s) {
    if (!s) return null;
    var d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  }

  var updatedAtDate = null;

  function tickClock() {
    if (updatedAtDate) {
      els.updatedAgo.textContent = "updated " + fmtAgo(Date.now() - updatedAtDate.getTime());
    } else {
      els.updatedAgo.textContent = "no data yet";
    }
    var connState = "ok";
    if (!lastFetchOkAt) connState = "down";
    else if (Date.now() - lastFetchOkAt > STALE_AFTER_MS) connState = "stale";
    els.connDot.className = "conn-dot" + (connState === "ok" ? "" : " " + connState);
  }

  // ---------- section renderers ----------

  function renderErrors(errors) {
    if (Array.isArray(errors) && errors.length > 0) {
      clear(els.errorList);
      errors.forEach(function (e) {
        els.errorList.appendChild(el("li", null, String(e)));
      });
      els.errorBanner.classList.remove("hidden");
    } else {
      els.errorBanner.classList.add("hidden");
    }
  }

  function renderFocus(focus) {
    focus = focus || {};
    text(els.focusPrimary, focus.primary);
    text(els.focusStatus, focus.status);

    var parked = Array.isArray(focus.parked) ? focus.parked : [];
    els.focusParkedSummary.textContent = "parked (" + parked.length + ")";
    clear(els.focusParkedList);
    if (parked.length === 0) {
      els.focusParkedList.appendChild(el("li", "empty-note", "nothing parked"));
    } else {
      parked.forEach(function (p) {
        els.focusParkedList.appendChild(el("li", null, String(p)));
      });
    }
  }

  function statusSlug(s) {
    return String(s || "unknown").trim().toLowerCase();
  }

  function renderCompute(compute) {
    compute = compute || {};
    text(els.computeKernel, compute.kernel);
    els.computeVersion.textContent =
      compute.version === null || compute.version === undefined ? "v—" : "v" + compute.version;

    var slug = statusSlug(compute.status);
    var statusClass = "status-unknown";
    if (slug.indexOf("run") !== -1) statusClass = "status-running";
    else if (slug.indexOf("complete") !== -1 || slug.indexOf("success") !== -1) statusClass = "status-complete";
    else if (slug.indexOf("error") !== -1 || slug.indexOf("cancel") !== -1 || slug.indexOf("fail") !== -1)
      statusClass = "status-error";
    els.computeStatus.className = "pill status-pill " + statusClass;
    text(els.computeStatus, compute.status ? compute.status.toUpperCase() : "UNKNOWN");

    var fp16 = compute.fp16;
    els.computeFp16.classList.remove("fp16-ok", "fp16-bad", "fp16-unknown");
    if (fp16 === true) {
      els.computeFp16.classList.add("fp16-ok");
      els.computeFp16.textContent = "fp16";
    } else if (fp16 === false) {
      els.computeFp16.classList.add("fp16-bad");
      els.computeFp16.textContent = "fp32!";
    } else {
      els.computeFp16.classList.add("fp16-unknown");
      els.computeFp16.textContent = "fp16?";
    }

    text(els.computeRunningFor, compute.running_for ? "running for " + compute.running_for : null);
    text(els.computeNote, compute.note || "");
  }

  var CLAIM_LANES = ["LIVE", "PROMISED", "VERIFIED", "POSTED", "RETRACTED"];

  function renderClaims(claims) {
    claims = claims || {};
    var counts = claims.counts || {};
    clear(els.claimsChips);
    CLAIM_LANES.forEach(function (lane) {
      var n = counts[lane];
      if (n === undefined || n === null) n = 0;
      var chip = el("span", "claim-chip lane-" + lane.toLowerCase());
      var label = el("span", null, lane + " ");
      var num = el("span", "n", String(n));
      chip.appendChild(label);
      chip.appendChild(num);
      els.claimsChips.appendChild(chip);
    });

    clear(els.claimsLive);
    var live = Array.isArray(claims.live) ? claims.live : [];
    if (live.length === 0) {
      els.claimsLive.appendChild(el("li", "empty-note", "none"));
    } else {
      live.forEach(function (c) {
        els.claimsLive.appendChild(el("li", null, String(c)));
      });
    }

    clear(els.claimsRetracted);
    var retracted = Array.isArray(claims.retracted) ? claims.retracted : [];
    if (retracted.length === 0) {
      els.claimsRetracted.appendChild(el("li", "empty-note", "none"));
    } else {
      retracted.forEach(function (c) {
        els.claimsRetracted.appendChild(el("li", null, String(c)));
      });
    }
  }

  var KB_TILE_ORDER = [
    ["nodes", "nodes"],
    ["papers", "papers"],
    ["sources", "sources"],
    ["topics", "topics"],
    ["claims", "claims"],
    ["verified", "verified"],
  ];

  function renderKb(kb) {
    kb = kb || {};
    clear(els.kbTiles);
    KB_TILE_ORDER.forEach(function (pair) {
      var key = pair[0], label = pair[1];
      var v = kb[key];
      var tile = el("div", "kb-tile");
      tile.appendChild(el("div", "val", v === null || v === undefined ? "—" : String(v)));
      tile.appendChild(el("div", "lbl", label));
      els.kbTiles.appendChild(tile);
    });
  }

  function renderLadder(ladder) {
    ladder = Array.isArray(ladder) ? ladder : [];
    clear(els.ladderList);
    if (ladder.length === 0) {
      els.ladderList.appendChild(el("li", "empty-note", "no ladder data"));
      return;
    }
    ladder.forEach(function (rung) {
      rung = rung || {};
      var status = statusSlug(rung.status);
      var item = el("li", "ladder-item " + (status === "done" || status === "active" || status === "todo" ? status : "todo"));
      var badge = el("span", "rung-badge", status === "done" ? "✓" : String(rung.rung !== undefined && rung.rung !== null ? rung.rung : "?"));
      var title = el("span", "rung-title", rung.title || "—");
      item.appendChild(badge);
      item.appendChild(title);
      els.ladderList.appendChild(item);
    });
  }

  function renderRepos(repos) {
    repos = Array.isArray(repos) ? repos : [];
    clear(els.reposTbody);
    if (repos.length === 0) {
      var tr = document.createElement("tr");
      var td = el("td", "empty-note", "no repo data");
      td.setAttribute("colspan", "5");
      tr.appendChild(td);
      els.reposTbody.appendChild(tr);
      return;
    }
    repos.forEach(function (r) {
      r = r || {};
      var tr = document.createElement("tr");

      var tdName = el("td", "repo-name", r.name || "—");
      var tdHead = el("td", "repo-head", r.head || "—");
      var tdSubject = el("td", "repo-subject", r.subject || "—");
      if (r.subject) tdSubject.title = String(r.subject);

      var tdDirty = document.createElement("td");
      var dot = el("span", "dirty-dot " + (r.dirty ? "dirty" : "clean"));
      dot.title = r.dirty ? "dirty" : "clean";
      tdDirty.appendChild(dot);

      var tdUnpushed = document.createElement("td");
      var n = r.unpushed;
      if (n === null || n === undefined) n = 0;
      var badge = el("span", "unpushed-badge" + (n > 0 ? " has-unpushed" : ""), n + " unpushed");
      tdUnpushed.appendChild(badge);

      tr.appendChild(tdName);
      tr.appendChild(tdHead);
      tr.appendChild(tdSubject);
      tr.appendChild(tdDirty);
      tr.appendChild(tdUnpushed);
      els.reposTbody.appendChild(tr);
    });
  }

  // ---------- top-level render ----------

  function render(data) {
    if (!data || typeof data !== "object") return;
    lastPayload = data;

    updatedAtDate = parseIso(data.updated_at);
    tickClock();

    renderErrors(data.errors);
    renderFocus(data.focus);
    renderCompute(data.compute);
    renderClaims(data.claims);
    renderKb(data.kb);
    renderLadder(data.ladder);
    renderRepos(data.repos);
  }

  // ---------- fetch loop ----------

  function fetchStatus() {
    fetch("/api/status", { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (data) {
        lastFetchOkAt = Date.now();
        render(data);
      })
      .catch(function (err) {
        // Network/parse failure: keep last-known render, surface a banner,
        // and let the conn dot degrade via tickClock().
        renderErrors(["frontend: /api/status unreachable (" + (err && err.message ? err.message : err) + ")"]);
      });
  }

  fetchStatus();
  setInterval(fetchStatus, POLL_MS);
  setInterval(tickClock, 1000);
})();

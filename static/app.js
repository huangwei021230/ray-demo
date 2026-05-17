/* ===================================================================
   Ray Distributed Computing — Interactive Paper Demo
   app.js — Complete client-side application
   =================================================================== */

class RayDemo {
  constructor() {
    this.ws = null;
    this.sessionId = this.generateSessionId();
    this.connected = false;
    this.reconnectTimer = null;

    // Server state
    this.state = null;
    this.programInfo = null;
    this.eventLogMap = {};
    this.maxStepSeen = 0;

    // UI state
    this.autoplaying = false;
    this.autoplayTimer = null;
    this.speed = 1000;
    this.detailExpanded = false;

    // Rendering state
    this.componentPositions = {};
    this.arrowAnimations = [];
    this.highlightElements = [];

    this.init();
  }

  generateSessionId() {
    return "sess-" + Math.random().toString(36).substring(2, 11) + "-" + Date.now().toString(36);
  }

  /* ================================================================
     INITIALIZATION
     ================================================================ */

  init() {
    this.setupControls();
    this.connect();
    this.renderEmptyState();
  }

  renderEmptyState() {
    this.clearArchitecture();
    this.clearTaskGraph();
  }

  /* ================================================================
     WEBSOCKET
     ================================================================ */

  connect() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = proto + "//" + window.location.host + "/ws/" + this.sessionId;
    try {
      this.ws = new WebSocket(url);
    } catch (e) {
      console.error("WebSocket creation failed:", e);
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.connected = true;
      this.updateConnectionStatus(true);
      const sel = document.getElementById("program-select");
      if (sel.value) {
        this.sendCommand({ action: "load", program: sel.value });
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (e) {
        console.error("Failed to parse message:", e);
      }
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.updateConnectionStatus(false);
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {};
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 3000);
  }

  sendCommand(cmd) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn("Cannot send - not connected");
      return;
    }
    this.ws.send(JSON.stringify(cmd));
  }

  /* ================================================================
     MESSAGE HANDLING
     ================================================================ */

  handleMessage(data) {
    switch (data.type) {
      case "program_loaded":
        this.handleProgramLoaded(data);
        break;
      case "state":
        this.handleStateUpdate(data);
        break;
      default:
        console.warn("Unknown message type:", data.type);
    }
  }

  handleProgramLoaded(data) {
    this.programInfo = data;
    this.eventLogMap = {};
    this.maxStepSeen = 0;
    this.stopAutoplay();

    document.getElementById("program-title").textContent = data.name || "";
    document.getElementById("total-steps").textContent = data.total_steps || 0;
    document.getElementById("current-step").textContent = "0";
    this.updateProgressBar(0, data.total_steps);

    const overview = document.getElementById("program-overview");
    const descEl = document.getElementById("program-description");
    const mapEl = document.getElementById("program-paper-mapping");
    if (overview && descEl && mapEl) {
      descEl.innerHTML = this.renderInlineMarkdown(data.description || "");
      mapEl.innerHTML = this.renderInlineMarkdown(data.paper_mapping || "");
      const hasContent = !!(data.description || data.paper_mapping);
      overview.classList.toggle("hidden", !hasContent);
      // Default to collapsed on every load so demos don't accidentally
      // cover the architecture area.
      overview.classList.add("collapsed");
      const toggle = document.getElementById("btn-overview-toggle");
      if (toggle) {
        toggle.setAttribute("aria-expanded", "false");
        const chev = toggle.querySelector(".overview-toggle-chevron");
        const hint = toggle.querySelector(".overview-toggle-hint");
        if (chev) chev.textContent = "▸";
        if (hint) hint.textContent = "点击展开";
      }
    }

    this.setControlsEnabled(true);
    this.clearArchitecture();
    this.clearTaskGraph();
    this.clearEventLog();
    this.clearArrows();
    this.clearHighlights();
  }

  handleStateUpdate(data) {
    this.state = data;

    document.getElementById("current-step").textContent = data.step_number;
    this.updateProgressBar(data.step_number, data.total_steps);

    if (data.event) {
      this.eventLogMap[data.step_number] = data.event;
      if (data.step_number > this.maxStepSeen) this.maxStepSeen = data.step_number;
      this.updateStepDescription(data.event);
    }

    this.renderArchitecture(data);
    this.renderTaskGraph(data.task_graph);
    this.renderEventLog(data.step_number);

    this.clearArrows();
    this.clearHighlights();

    requestAnimationFrame(() => {
      if (data.event) {
        if (data.event.arrows && data.event.arrows.length) {
          this.animateArrows(data.event.arrows);
        }
        if (data.event.highlights && data.event.highlights.length) {
          this.applyHighlights(data.event.highlights);
        }
      }
    });

    if (data.step_number >= data.total_steps && this.autoplaying) {
      this.stopAutoplay();
    }
  }

  /* ================================================================
     CONTROLS
     ================================================================ */

  setupControls() {
    document.getElementById("program-select").addEventListener("change", (e) => {
      if (e.target.value) this.loadProgram(e.target.value);
    });

    document.getElementById("btn-step").addEventListener("click", () => this.stepForward());
    document.getElementById("btn-back").addEventListener("click", () => this.stepBack());
    document.getElementById("btn-reset").addEventListener("click", () => this.reset());
    document.getElementById("btn-play").addEventListener("click", () => this.toggleAutoplay());

    const slider = document.getElementById("speed-slider");
    slider.addEventListener("input", (e) => {
      this.speed = parseInt(e.target.value);
      document.getElementById("speed-value").textContent =
        this.speed >= 1000 ? (this.speed / 1000).toFixed(1) + "s" : this.speed + "ms";
      if (this.autoplaying) {
        this.stopAutoplay();
        this.startAutoplay();
      }
    });

    document.getElementById("progress-clickable").addEventListener("click", (e) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const totalSteps = parseInt(document.getElementById("total-steps").textContent) || 0;
      if (totalSteps > 0) {
        const step = Math.max(1, Math.round(ratio * totalSteps));
        this.gotoStep(step);
      }
    });

    document.getElementById("btn-detail-toggle").addEventListener("click", () => {
      this.detailExpanded = !this.detailExpanded;
      const detail = document.getElementById("step-detail");
      const btn = document.getElementById("btn-detail-toggle");
      detail.classList.toggle("expanded", this.detailExpanded);
      btn.textContent = this.detailExpanded ? "\u25BE Details" : "\u25B8 Details";
    });

    const overviewToggle = document.getElementById("btn-overview-toggle");
    if (overviewToggle) {
      overviewToggle.addEventListener("click", () => {
        const overview = document.getElementById("program-overview");
        if (!overview) return;
        const isCollapsed = overview.classList.toggle("collapsed");
        overviewToggle.setAttribute("aria-expanded", String(!isCollapsed));
        overviewToggle.querySelector(".overview-toggle-chevron").textContent =
          isCollapsed ? "\u25B8" : "\u25BE";
        overviewToggle.querySelector(".overview-toggle-hint").textContent =
          isCollapsed ? "\u70B9\u51FB\u5C55\u5F00" : "\u70B9\u51FB\u6536\u8D77";
      });
    }

    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
      switch (e.key) {
        case "ArrowRight": e.preventDefault(); this.stepForward(); break;
        case "ArrowLeft":  e.preventDefault(); this.stepBack(); break;
        case " ":          e.preventDefault(); this.toggleAutoplay(); break;
        case "r":          e.preventDefault(); this.reset(); break;
      }
    });
  }

  loadProgram(name) {
    this.eventLogMap = {};
    this.maxStepSeen = 0;
    this.stopAutoplay();
    this.clearArrows();
    this.clearHighlights();
    this.sendCommand({ action: "load", program: name });
  }

  /**
   * Render a tiny subset of Markdown safely:
   *   **bold**, __bold__, *italic*, _italic_, `code`.
   * Everything else is HTML-escaped. No links, no HTML passthrough.
   */
  renderInlineMarkdown(src) {
    if (!src) return "";
    var esc = String(src)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
    // Bold first (longer markers), then italic, then code.
    esc = esc.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    esc = esc.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
    esc = esc.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    esc = esc.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
    esc = esc.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    // Line breaks: blank line → paragraph break, single \n → soft break.
    esc = esc.replace(/\n{2,}/g, "<br><br>").replace(/\n/g, "<br>");
    return esc;
  }

  async fetchPrograms() {
    try {
      const res = await fetch("/api/programs");
      const programs = await res.json();
      const sel = document.getElementById("program-select");
      sel.innerHTML = '<option value="">Select program…</option>';
      programs.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.name;
        opt.title = p.description || "";
        sel.appendChild(opt);
      });
    } catch (err) {
      console.error("Failed to load programs:", err);
    }
  }

  stepForward()  { this.sendCommand({ action: "step" }); }
  stepBack()     { this.sendCommand({ action: "back" }); }

  reset() {
    this.stopAutoplay();
    this.eventLogMap = {};
    this.maxStepSeen = 0;
    this.clearEventLog();
    this.clearArrows();
    this.clearHighlights();
    this.sendCommand({ action: "reset" });
  }

  gotoStep(n) { this.sendCommand({ action: "goto", step: n }); }

  toggleAutoplay() {
    this.autoplaying ? this.stopAutoplay() : this.startAutoplay();
  }

  startAutoplay() {
    this.autoplaying = true;
    document.getElementById("btn-play").classList.add("active");
    // Drive playback entirely from the client so a pause click takes effect
    // immediately. (The server has no blocking autoplay loop to interrupt.)
    this.autoplayTimer = setInterval(() => this.stepForward(), this.speed);
  }

  stopAutoplay() {
    this.autoplaying = false;
    document.getElementById("btn-play").classList.remove("active");
    if (this.autoplayTimer) {
      clearInterval(this.autoplayTimer);
      this.autoplayTimer = null;
    }
  }

  setControlsEnabled(enabled) {
    ["btn-step", "btn-back", "btn-reset", "btn-play"].forEach(id => {
      document.getElementById(id).disabled = !enabled;
    });
  }

  /* ================================================================
     UI UPDATES
     ================================================================ */

  updateConnectionStatus(connected) {
    const el = document.getElementById("connection-status");
    el.textContent = connected ? "Connected" : "Disconnected";
    el.className = "status-dot " + (connected ? "status-connected" : "status-disconnected");
  }

  updateProgressBar(current, total) {
    document.getElementById("progress-fill").style.width =
      total > 0 ? ((current / total) * 100) + "%" : "0%";
  }

  updateStepDescription(event) {
    document.getElementById("step-description").classList.remove("empty");
    const badge = document.getElementById("phase-badge");
    badge.textContent = (event.phase || "\u2014").replace(/_/g, " ");
    badge.className = event.phase || "";
    badge.id = "phase-badge";

    document.getElementById("step-desc-text").textContent = event.description || "";
    document.getElementById("step-detail-text").textContent = event.detail || "";
  }

  /* ================================================================
     SVG UTILITIES
     ================================================================ */

  svgEl(tag, attrs) {
    if (!attrs) attrs = {};
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    const keys = Object.keys(attrs);
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const v = attrs[k];
      if (v !== undefined && v !== null) el.setAttribute(k, v);
    }
    return el;
  }

  svgText(text, x, y, attrs) {
    if (!attrs) attrs = {};
    const el = this.svgEl("text", Object.assign({ x: x, y: y }, attrs));
    el.textContent = text;
    return el;
  }

  _measureTextWidth(text) {
    if (!this._measureSpan) {
      this._measureSpan = document.createElement("span");
      this._measureSpan.style.fontFamily = "'IBM Plex Mono', monospace";
      this._measureSpan.style.fontSize = "12px";
      this._measureSpan.style.fontWeight = "600";
      this._measureSpan.style.position = "absolute";
      this._measureSpan.style.visibility = "hidden";
      this._measureSpan.style.whiteSpace = "pre";
      document.body.appendChild(this._measureSpan);
    }
    this._measureSpan.textContent = text || "";
    return this._measureSpan.offsetWidth;
  }

  /* ================================================================
     ARCHITECTURE RENDERING
     ================================================================ */

  clearArchitecture() {
    const svg = document.getElementById("arch-svg");
    const defs = svg.querySelector("defs");
    svg.innerHTML = "";
    if (defs) svg.appendChild(defs);
    this.componentPositions = {};
  }

  renderArchitecture(data) {
    const svg = document.getElementById("arch-svg");
    const defs = svg.querySelector("defs");
    svg.innerHTML = "";
    if (defs) svg.appendChild(defs);
    this.componentPositions = {};

    if (!data || !data.nodes || Object.keys(data.nodes).length === 0) {
      svg.setAttribute("viewBox", "0 0 1000 200");
      svg.appendChild(this.svgText("Load an example to visualize the Ray cluster", 500, 100, {
        "text-anchor": "middle", fill: "#9ca3af",
        "font-family": "IBM Plex Sans", "font-size": "18"
      }));
      return;
    }

    const nodes = Object.values(data.nodes);
    const numNodes = nodes.length;
    const gcs = data.gcs || {};
    const globalQueue = data.global_scheduler_queue || [];

    const W = 1200;
    const GCS_MAX_ROWS = 10;

    // GCS layout
    const gcsX = 15, gcsY = 8, gcsW = 320;
    const gcsInnerPad = 8;
    const gcsTableH = this._calcGCSTableHeight(gcs, GCS_MAX_ROWS);
    const gcsH = 68 + gcsTableH;

    // Global Scheduler layout
    const gsW = 240, gsH = 80;
    const gsX = Math.min(430, gcsX + gcsW + 30);
    const gsY = 14;

    // Nodes layout
    const nodeGap = 16;
    const nodeTopY = Math.max(gcsY + gcsH + 30, 220);
    const maxNodeW = 280, minNodeW = 140;
    const nodeW = Math.max(minNodeW, Math.min(maxNodeW,
      (W - 30 - (numNodes - 1) * nodeGap) / numNodes));

    let maxNodeH = 0;
    for (let i = 0; i < nodes.length; i++) {
      const nh = this._calcNodeHeight(nodes[i], nodeW);
      if (nh > maxNodeH) maxNodeH = nh;
    }
    const nodeH = maxNodeH;

    const totalNodeW = numNodes * nodeW + (numNodes - 1) * nodeGap;
    const nodeStartX = (W - totalNodeW) / 2;

    const totalH = nodeTopY + nodeH + 12;
    svg.setAttribute("viewBox", "0 0 " + W + " " + totalH);

    this._drawBgConnections(svg, gsX, gsY, gsW, gsH, gcsX, gcsY, gcsW, gcsH,
      nodeStartX, nodeTopY, nodeW, numNodes, nodeGap);
    this._drawGCS(svg, gcs, gcsX, gcsY, gcsW, gcsH, gcsInnerPad, GCS_MAX_ROWS);
    this._drawGlobalScheduler(svg, globalQueue, gsX, gsY, gsW, gsH);
    for (let i = 0; i < nodes.length; i++) {
      this._drawNode(svg, nodes[i], nodeStartX + i * (nodeW + nodeGap), nodeTopY, nodeW, nodeH);
    }
  }

  _calcGCSTableHeight(gcs, maxRows) {
    let h = 0;
    const tableKeys = ["object_table", "task_table", "function_table", "actor_table"];
    for (let ti = 0; ti < tableKeys.length; ti++) {
      const key = tableKeys[ti];
      const n = gcs[key] ? Object.keys(gcs[key]).length : 0;
      const capped = Math.min(n, maxRows);
      h += 22 + Math.max(capped, 1) * 18 + 5 + 5;
    }
    return h;
  }

  _calcNodeHeight(node, w) {
    const px = 6;
    let cy = px + 18 + 12 + 8 + 6;
    if (node.is_driver) cy += 28;
    const workers = node.workers || [];
    cy += 18 + Math.max(workers.length, 0) * 18 + 3 + 3;
    cy += 36 + 3;
    const objs = node.object_store ? Object.entries(node.object_store) : [];
    const maxShow = Math.min(objs.length, 10);
    const show = Math.max(1, Math.min(objs.length, maxShow));
    cy += 22 + show * 20 + 5 + 3;
    const actors = node.actors || [];
    if (actors.length > 0) cy += 22 + actors.length * 20 + 5 + 3;
    return cy;
  }

  _drawGCS(svg, gcs, x, y, w, h, pad, maxRows) {
    const g = this.svgEl("g", { id: "GCS" });
    this.componentPositions["GCS"] = { x: x, y: y, width: w, height: h };

    g.appendChild(this.svgEl("rect", {
      x: x + 3, y: y + 3, width: w, height: h,
      fill: "rgba(0,0,0,0.06)", rx: "8", ry: "8"
    }));
    g.appendChild(this.svgEl("rect", { x: x, y: y, width: w, height: h, class: "arch-box-gcs" }));

    g.appendChild(this.svgText("GCS", x + pad + 4, y + 28, { class: "arch-title", "font-size": "11" }));
    g.appendChild(this.svgText("Global Control Store", x + pad + 4, y + 48, {
      class: "arch-subtitle", "font-size": "8", fill: "#546e7a"
    }));
    g.appendChild(this.svgEl("line", {
      x1: x + pad, y1: y + 58, x2: x + w - pad, y2: y + 58, class: "arch-divider"
    }));

    var ty = y + 68;
    // Width budget for a single row (in characters) \u2014 keep entries from
    // overflowing the GCS box. SVG text is non-monospace so this is approximate.
    var maxChars = Math.max(20, Math.floor((w - pad * 2 - 12) / 5.2));
    function clip(s) {
      return s.length > maxChars ? s.slice(0, maxChars - 1) + "\u2026" : s;
    }
    var tables = [
      { key: "object_table", label: "Object Table", id: "GCS_object_table",
        fmt: function(k, v) { return clip(k + " \u2192 " + v.location + " (" + v.size + "B)"); } },
      { key: "task_table", label: "Task Table", id: "GCS_task_table",
        fmt: function(k, v) { return clip(k + ": " + v.status + (v.node ? " @ " + v.node : "")); } },
      { key: "function_table", label: "Function Table", id: "GCS_function_table",
        fmt: function(k, v) { return clip(k + " (" + v.num_returns + " rets)"); } },
      { key: "actor_table", label: "Actor Table", id: "GCS_actor_table",
        fmt: function(k, v) {
          if (typeof v !== "object" || v === null) return clip(k + ": " + v);
          var cls = v.class_name || "?";
          var node = v.node || "\u2014";
          var methods = Array.isArray(v.methods) && v.methods.length
              ? " [" + v.methods.join(",") + "]" : "";
          return clip(k + " \u2192 " + cls + "@" + node + methods);
        } }
    ];

    for (var ti = 0; ti < tables.length; ti++) {
      var t = tables[ti];
      var entries = gcs[t.key] ? Object.entries(gcs[t.key]) : [];
      var rowH = 18;
      var displayRows = Math.min(entries.length, maxRows);
      var th = 22 + Math.max(displayRows, 1) * rowH + 5;

      var tg = this.svgEl("g", { id: t.id });
      this.componentPositions[t.id] = { x: x + pad, y: ty, width: w - pad * 2, height: th };

      tg.appendChild(this.svgEl("rect", {
        x: x + pad, y: ty, width: w - pad * 2, height: th,
        fill: "#ffffff", stroke: "#e0e0e0", "stroke-width": "0.5", rx: "3", ry: "3"
      }));
      tg.appendChild(this.svgText(t.label, x + pad + 4, ty + 14, { class: "arch-section-header" }));

      if (entries.length === 0) {
        tg.appendChild(this.svgText("(empty)", x + pad + 4, ty + 30, {
          class: "arch-value", fill: "#b0b0b0", "font-style": "italic"
        }));
      } else {
        for (var ei = 0; ei < displayRows; ei++) {
          var entry = entries[ei];
          tg.appendChild(this.svgText(t.fmt(entry[0], entry[1]), x + pad + 10, ty + 30 + ei * rowH, {
            class: "arch-value"
          }));
        }
        if (entries.length > displayRows) {
          tg.appendChild(this.svgText("\u2026 +" + (entries.length - displayRows) + " more", x + pad + 6, ty + 30 + displayRows * rowH, {
            class: "arch-value", fill: "#9ca3af", "font-style": "italic"
          }));
        }
      }
      g.appendChild(tg);
      ty += th + 4;
    }

    svg.appendChild(g);
  }

  _drawGlobalScheduler(svg, queue, x, y, w, h) {
    const g = this.svgEl("g", { id: "global_scheduler" });
    this.componentPositions["global_scheduler"] = { x: x, y: y, width: w, height: h };

    g.appendChild(this.svgEl("rect", {
      x: x + 3, y: y + 3, width: w, height: h,
      fill: "rgba(0,0,0,0.06)", rx: "8", ry: "8"
    }));
    g.appendChild(this.svgEl("rect", { x: x, y: y, width: w, height: h, class: "arch-box-scheduler" }));

    g.appendChild(this.svgText("Global Scheduler", x + 10, y + 28, {
      class: "arch-title", "font-size": "11"
    }));

    var qLen = queue ? queue.length : 0;
    if (qLen > 0) {
      g.appendChild(this.svgText("Queue: " + qLen + " task" + (qLen > 1 ? "s" : ""), x + 10, y + 52, {
        class: "arch-subtitle", "font-size": "10"
      }));
      var preview = queue.slice(0, 10).join(", ") + (qLen > 10 ? "\u2026" : "");
      g.appendChild(this.svgText(preview, x + 10, y + 70, { class: "arch-value" }));
    } else {
      g.appendChild(this.svgText("Queue: empty", x + 10, y + 52, {
        class: "arch-subtitle", "font-size": "10"
      }));
    }

    svg.appendChild(g);
  }

  _drawNode(svg, node, x, y, w, h) {
    var nid = node.node_id;
    var g = this.svgEl("g", { id: nid });
    this.componentPositions[nid] = { x: x, y: y, width: w, height: h };

    var isDriver = node.is_driver;
    var isDead = node.is_dead;
    var boxClass = isDead ? "arch-box-dead" : (isDriver ? "arch-box-driver" : "arch-box-node");
    if (isDead) g.setAttribute("class", "node-dead");

    g.appendChild(this.svgEl("rect", {
      x: x + 2, y: y + 2, width: w, height: h,
      fill: "rgba(0,0,0,0.05)", rx: "6", ry: "6"
    }));
    g.appendChild(this.svgEl("rect", { x: x, y: y, width: w, height: h, class: boxClass }));

    var px = 6;
    var cy = y + px + 18;

    g.appendChild(this.svgText(nid, x + px + 2, cy, { class: "arch-title", "font-size": "11" }));
    var roleText = isDead ? "\u2620 DEAD" : (isDriver ? "\u2605 Driver" : "Worker");
    var roleColor = isDead ? "#c62828" : (isDriver ? "#2e7d32" : "#546e7a");
    g.appendChild(this.svgText(roleText, x + w - px - 2, cy, {
      class: "arch-subtitle", "font-size": "9", "text-anchor": "end", fill: roleColor
    }));
    cy += 12;

    g.appendChild(this.svgEl("line", {
      x1: x + px, y1: cy, x2: x + w - px, y2: cy, class: "arch-divider"
    }));
    cy += 8;

    // Driver subcomponent
    if (isDriver) {
      var dg = this.svgEl("g", { id: nid + "_driver" });
      this.componentPositions[nid + "_driver"] = { x: x + px, y: cy, width: w - px * 2, height: 20 };
      dg.appendChild(this.svgText("\u25CF Driver Process", x + px + 4, cy + 13, {
        class: "arch-value", fill: "#2e7d32", "font-weight": "600", "font-size": "9"
      }));
      g.appendChild(dg);
      cy += 28;
    }

    // Workers
    var workers = node.workers || [];
    var workerTasks = node.worker_tasks || {};
    var wH = 18 + Math.max(workers.length, 0) * 18 + 3;
    var wg = this.svgEl("g", { id: nid + "_worker" });
    this.componentPositions[nid + "_worker"] = { x: x + px, y: cy, width: w - px * 2, height: wH };

    wg.appendChild(this.svgText("Workers (" + workers.length + ")", x + px + 4, cy + 13, {
      class: "arch-section-header", "font-size": "8"
    }));
    for (var wi = 0; wi < workers.length; wi++) {
      var wShort = workers[wi].split("_").pop();
      var wTask = workerTasks[workers[wi]];
      var wLabel = wShort + (wTask ? " [" + wTask + "]" : " [idle]");
      var wColor = wTask ? "#e65100" : "#6b7280";
      wg.appendChild(this.svgText(wLabel, x + px + 12, cy + 30 + wi * 18, {
        class: "arch-value", "font-size": "7.5", fill: wColor
      }));
    }

    // Workers
    var workers = node.workers || [];
    var workerTasks = node.worker_tasks || {};
    var wH = 14 + Math.max(workers.length, 0) * 14 + 2;
    var wg = this.svgEl("g", { id: nid + "_worker" });
    this.componentPositions[nid + "_worker"] = { x: x + px, y: cy, width: w - px * 2, height: wH };

    wg.appendChild(this.svgText("Workers (" + workers.length + ")", x + px + 4, cy + 10, {
      class: "arch-section-header", "font-size": "8"
    }));
    for (var wi = 0; wi < workers.length; wi++) {
      var wShort = workers[wi].split("_").pop();
      var wTask = workerTasks[workers[wi]];
      var wLabel = wShort + (wTask ? " [" + wTask + "]" : " [idle]");
      var wColor = wTask ? "#e65100" : "#6b7280";
      wg.appendChild(this.svgText(wLabel, x + px + 12, cy + 24 + wi * 14, {
        class: "arch-value", "font-size": "7.5", fill: wColor
      }));
    }
    g.appendChild(wg);
    cy += wH + 3;

    // Local Scheduler
    var lsQueue = node.local_queue || [];
    var lsH = 36;
    var lsg = this.svgEl("g", { id: nid + "_local_scheduler" });
    this.componentPositions[nid + "_local_scheduler"] = { x: x + px, y: cy, width: w - px * 2, height: lsH };

    lsg.appendChild(this.svgEl("rect", {
      x: x + px, y: cy, width: w - px * 2, height: lsH,
      fill: "#f5f5f5", stroke: "#e0e0e0", "stroke-width": "0.5", rx: "3", ry: "3"
    }));
    lsg.appendChild(this.svgText("Local Scheduler", x + px + 4, cy + 16, {
      class: "arch-section-header", "font-size": "8"
    }));
    lsg.appendChild(this.svgText("queue: " + lsQueue.length, x + px + 4, cy + 32, {
      class: "arch-value", "font-size": "8"
    }));
    if (lsQueue.length > 0) {
      var lsPreview = lsQueue.slice(0, 10).join(", ") + (lsQueue.length > 10 ? "\u2026" : "");
      lsg.appendChild(this.svgText(lsPreview, x + px + 100, cy + 32, {
        class: "arch-value", "font-size": "7.5", fill: "#6b7280"
      }));
    }
    g.appendChild(lsg);
    cy += lsH + 3;

    // Object Store
    var objs = node.object_store ? Object.entries(node.object_store) : [];
    var maxShow = Math.max(1, Math.floor((y + h - cy - 30) / 20));
    var show = Math.max(1, Math.min(objs.length, maxShow));
    var oH = 22 + show * 20 + 5;
    var og = this.svgEl("g", { id: nid + "_object_store" });
    this.componentPositions[nid + "_object_store"] = { x: x + px, y: cy, width: w - px * 2, height: oH };

    og.appendChild(this.svgText("Object Store (" + objs.length + ")", x + px + 4, cy + 16, {
      class: "arch-section-header", "font-size": "8"
    }));
    if (objs.length === 0) {
      og.appendChild(this.svgText("(empty)", x + px + 12, cy + 32, {
        class: "arch-value", "font-size": "7.5", fill: "#b0b0b0", "font-style": "italic"
      }));
    } else {
      for (var oi = 0; oi < show; oi++) {
        og.appendChild(this.svgText(objs[oi][0] + ": " + objs[oi][1], x + px + 12, cy + 32 + oi * 20, {
          class: "arch-value", "font-size": "7.5"
        }));
      }
      if (objs.length > show) {
        og.appendChild(this.svgText("+" + (objs.length - show) + " more", x + px + 12, cy + 32 + show * 20, {
          class: "arch-value", "font-size": "7.5", fill: "#9ca3af", "font-style": "italic"
        }));
      }
    }
    g.appendChild(og);
    cy += oH + 3;

    // Actors
    var actors = node.actors || [];
    if (actors.length > 0) {
      var aH = 22 + actors.length * 20 + 5;
      var ag = this.svgEl("g", { id: nid + "_actor" });
      this.componentPositions[nid + "_actor"] = { x: x + px, y: cy, width: w - px * 2, height: aH };

      ag.appendChild(this.svgText("Actors (" + actors.length + ")", x + px + 4, cy + 16, {
        class: "arch-section-header", "font-size": "8"
      }));
      for (var ai = 0; ai < actors.length; ai++) {
        ag.appendChild(this.svgText(actors[ai], x + px + 12, cy + 32 + ai * 20, {
          class: "arch-value", "font-size": "7.5"
        }));
      }
      g.appendChild(ag);
    } else {
      this.componentPositions[nid + "_actor"] = { x: x + px, y: cy, width: w - px * 2, height: 14 };
    }

    svg.appendChild(g);
  }

  _drawBgConnections(svg, gsX, gsY, gsW, gsH, gcsX, gcsY, gcsW, gcsH,
                      nodeStartX, nodeTopY, nodeW, numNodes, nodeGap) {
    var g = this.svgEl("g", { class: "bg-connections", opacity: "0.12" });
    var gsCx = gsX + gsW / 2;
    var gsBottom = gsY + gsH;

    for (var i = 0; i < numNodes; i++) {
      var nx = nodeStartX + i * (nodeW + nodeGap) + nodeW / 2;
      g.appendChild(this.svgEl("path", {
        d: "M " + gsCx + " " + (gsBottom + 8) + " C " + gsCx + " " + (gsBottom + 60) + ", " + nx + " " + (nodeTopY - 60) + ", " + nx + " " + (nodeTopY - 5),
        stroke: "#78909c", "stroke-width": "1", "stroke-dasharray": "6 4", fill: "none"
      }));
    }

    g.appendChild(this.svgEl("line", {
      x1: gcsX + gcsW + 5, y1: gcsY + gcsH / 2,
      x2: gsX - 5, y2: gsY + gsH / 2,
      stroke: "#78909c", "stroke-width": "1", "stroke-dasharray": "6 4"
    }));

    for (var j = 0; j < numNodes; j++) {
      var nx2 = nodeStartX + j * (nodeW + nodeGap) + nodeW / 2;
      g.appendChild(this.svgEl("path", {
        d: "M " + (gcsX + gcsW / 2) + " " + (gcsY + gcsH + 5) + " C " + (gcsX + gcsW / 2) + " " + (gcsY + gcsH + 80) + ", " + nx2 + " " + (nodeTopY - 40) + ", " + nx2 + " " + (nodeTopY - 5),
        stroke: "#78909c", "stroke-width": "0.7", "stroke-dasharray": "4 6", fill: "none"
      }));
    }

    svg.insertBefore(g, svg.querySelector("defs").nextSibling);
  }

  /* ================================================================
     ARROW ANIMATION
     ================================================================ */

  clearArrows() {
    for (var i = 0; i < this.arrowAnimations.length; i++) {
      if (this.arrowAnimations[i].raf) cancelAnimationFrame(this.arrowAnimations[i].raf);
    }
    this.arrowAnimations = [];
    var svg = document.getElementById("arch-svg");
    if (svg) svg.querySelectorAll(".arrow-layer").forEach(function(el) { el.remove(); });
  }

  animateArrows(arrows) {
    if (!arrows || !arrows.length) return;
    var svg = document.getElementById("arch-svg");
    if (!svg) return;

    var layer = this.svgEl("g", { class: "arrow-layer" });
    svg.appendChild(layer);

    var self = this;
    arrows.forEach(function(arrow, idx) {
      self._animateOneArrow(layer, arrow, idx * 120);
    });
  }

  _animateOneArrow(parent, arrow, delay) {
    var fromPos = this._getComponentCenter(arrow.from);
    var toPos = this._getComponentCenter(arrow.to);
    if (!fromPos || !toPos) {
      console.warn("Arrow target not found:", arrow.from, "\u2192", arrow.to);
      return;
    }

    var fromRect = this.componentPositions[arrow.from];
    var toRect = this.componentPositions[arrow.to];
    var startPt = fromRect ? this._edgePoint(fromRect, toPos) : fromPos;
    var endPt = toRect ? this._edgePoint(toRect, fromPos) : toPos;

    var dx = endPt.x - startPt.x;
    var dy = endPt.y - startPt.y;
    var dist = Math.sqrt(dx * dx + dy * dy) || 1;
    var midX = (startPt.x + endPt.x) / 2;
    var midY = (startPt.y + endPt.y) / 2;
    var curveAmt = Math.min(40, dist * 0.18);
    var cpX = midX + (-dy / dist) * curveAmt;
    var cpY = midY + (dx / dist) * curveAmt;

    var pathD = "M " + startPt.x + " " + startPt.y + " Q " + cpX + " " + cpY + " " + endPt.x + " " + endPt.y;

    var aType = arrow.type || "data";

    var pathAttrs = { d: pathD, class: "arrow-path-" + aType, opacity: "0" };
    if (arrow.style === "dashed") pathAttrs["stroke-dasharray"] = "8 4";
    var pathEl = this.svgEl("path", pathAttrs);
    parent.appendChild(pathEl);

    var dotEl = this.svgEl("circle", {
      cx: startPt.x, cy: startPt.y, r: "5",
      class: "arrow-dot-" + aType, opacity: "0"
    });
    parent.appendChild(dotEl);

    var labelBg, labelEl;
    if (arrow.label) {
      labelBg = this.svgEl("rect", { class: "arrow-label-bg", opacity: "0" });
      labelEl = this.svgText(arrow.label, cpX, cpY - 10, {
        class: "arrow-label", "text-anchor": "middle", opacity: "0"
      });
      parent.appendChild(labelBg);
      parent.appendChild(labelEl);
    }

    var anim = { raf: null };
    this.arrowAnimations.push(anim);

    var t0 = performance.now() + delay;
    var fadeIn = 200;
    var travel = Math.max(500, dist * 1.0);
    var hold = 600;
    var fadeOut = 500;
    var totalMs = fadeIn + travel + hold + fadeOut;

    var tick = function(now) {
      var elapsed = now - t0;
      if (elapsed < 0) { anim.raf = requestAnimationFrame(tick); return; }

      var pathOp = 0, dotOp = 0, dotCx = startPt.x, dotCy = startPt.y;

      if (elapsed < fadeIn) {
        var p = elapsed / fadeIn;
        pathOp = p; dotOp = p;
      } else if (elapsed < fadeIn + travel) {
        pathOp = 1; dotOp = 1;
        var p2 = (elapsed - fadeIn) / travel;
        try {
          var len = pathEl.getTotalLength();
          var pt = pathEl.getPointAtLength(p2 * len);
          dotCx = pt.x; dotCy = pt.y;
        } catch (_) {
          dotCx = startPt.x + (endPt.x - startPt.x) * p2;
          dotCy = startPt.y + (endPt.y - startPt.y) * p2;
        }
      } else if (elapsed < fadeIn + travel + hold) {
        pathOp = 1; dotOp = 0;
      } else if (elapsed < totalMs) {
        var p3 = 1 - (elapsed - fadeIn - travel - hold) / fadeOut;
        pathOp = Math.max(0, p3) * 0.5;
        dotOp = 0;
      } else {
        pathEl.setAttribute("opacity", "0.25");
        dotEl.setAttribute("opacity", "0");
        if (labelEl) labelEl.setAttribute("opacity", "0.4");
        if (labelBg) labelBg.setAttribute("opacity", "0.2");
        return;
      }

      pathEl.setAttribute("opacity", pathOp);
      dotEl.setAttribute("opacity", dotOp);
      dotEl.setAttribute("cx", dotCx);
      dotEl.setAttribute("cy", dotCy);
      if (labelEl) labelEl.setAttribute("opacity", pathOp);
      if (labelBg) {
        labelBg.setAttribute("opacity", pathOp * 0.85);
        try {
          var bb = labelEl.getBBox();
          labelBg.setAttribute("x", bb.x - 3);
          labelBg.setAttribute("y", bb.y - 1);
          labelBg.setAttribute("width", bb.width + 6);
          labelBg.setAttribute("height", bb.height + 2);
        } catch (_) {}
      }

      anim.raf = requestAnimationFrame(tick);
    };

    anim.raf = requestAnimationFrame(tick);
  }

  _getComponentCenter(id) {
    if (this.componentPositions[id]) {
      var p = this.componentPositions[id];
      return { x: p.x + p.width / 2, y: p.y + p.height / 2 };
    }
    var keys = Object.keys(this.componentPositions);
    for (var i = 0; i < keys.length; i++) {
      if (id.startsWith(keys[i] + "_") || keys[i].startsWith(id + "_")) {
        var p2 = this.componentPositions[keys[i]];
        return { x: p2.x + p2.width / 2, y: p2.y + p2.height / 2 };
      }
    }
    return null;
  }

  _edgePoint(rect, target) {
    var cx = rect.x + rect.width / 2;
    var cy = rect.y + rect.height / 2;
    var dx = target.x - cx;
    var dy = target.y - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };

    var hw = rect.width / 2 + 4;
    var hh = rect.height / 2 + 4;

    var ex, ey;
    if (Math.abs(dx) * hh > Math.abs(dy) * hw) {
      ex = dx > 0 ? rect.x + rect.width + 4 : rect.x - 4;
      ey = cy + dy * (ex - cx) / dx;
    } else {
      ey = dy > 0 ? rect.y + rect.height + 4 : rect.y - 4;
      ex = cx + dx * (ey - cy) / dy;
    }
    return { x: ex, y: ey };
  }

  /* ================================================================
     HIGHLIGHTS
     ================================================================ */

  clearHighlights() {
    for (var i = 0; i < this.highlightElements.length; i++) {
      this.highlightElements[i].classList.remove("highlight-active", "highlight-new", "highlight-modified");
    }
    this.highlightElements = [];
  }

  applyHighlights(highlights) {
    if (!highlights) return;
    for (var i = 0; i < highlights.length; i++) {
      var h = highlights[i];
      var el = document.getElementById(h.id);
      if (el) {
        el.classList.add("highlight-" + (h.type || "active"));
        this.highlightElements.push(el);
      } else {
        var svg = document.getElementById("arch-svg");
        if (svg) {
          var found = svg.querySelector('[id="' + h.id + '"]');
          if (found) {
            found.classList.add("highlight-" + (h.type || "active"));
            this.highlightElements.push(found);
          }
        }
      }
    }
  }

  /* ================================================================
     TASK GRAPH RENDERING — compact sizing
     ================================================================ */

  // Task graph node dimensions (compact)
  static get TG_DATA_R()      { return 12; }
  static get TG_TASK_W()      { return 54; }
  static get TG_TASK_H()      { return 20; }
  static get TG_H_SPACING()   { return 70; }
  static get TG_V_SPACING()   { return 55; }

  clearTaskGraph() {
    var svg = document.getElementById("task-graph-svg");
    svg.innerHTML = "";
    svg.removeAttribute("viewBox");
  }

  renderTaskGraph(graph) {
    var svg = document.getElementById("task-graph-svg");
    svg.innerHTML = "";

    if (!graph || !graph.nodes || Object.keys(graph.nodes).length === 0) {
      svg.setAttribute("viewBox", "0 0 300 50");
      svg.appendChild(this.svgText("Task graph will appear here", 150, 25, {
        "text-anchor": "middle", fill: "#9ca3af",
        "font-family": "IBM Plex Sans", "font-size": "13"
      }));
      return;
    }

    var layout = this._layoutTaskGraph(graph);
    var positions = layout.positions;
    var width = layout.width;
    var height = layout.height;
    var offsetX = layout.offsetX;
    var offsetY = layout.offsetY;

    var padX = 16, padY = 12;
    svg.setAttribute("viewBox", "0 0 " + (width + padX * 2) + " " + (height + padY * 2));

    var g = this.svgEl("g", { transform: "translate(" + (padX + offsetX) + ", " + (padY + offsetY) + ")" });

    this._ensureTaskGraphDefs(svg);

    // Edges behind nodes
    if (graph.edges) {
      for (var ei = 0; ei < graph.edges.length; ei++) {
        var edge = graph.edges[ei];
        var fromP = positions[edge.from];
        var toP = positions[edge.to];
        if (!fromP || !toP) continue;

        var fromNode = graph.nodes[edge.from];
        var toNode = graph.nodes[edge.to];
        var s = this._tgEdgePoint(fromP, fromNode, toP);
        var e = this._tgEdgePoint(toP, toNode, fromP);

        var cls = "tg-edge-" + (edge.type || "data");
        var marker = edge.type === "control" ? "url(#tg-arrow-control)" :
                     edge.type === "stateful" ? "url(#tg-arrow-stateful)" :
                     "url(#tg-arrow-data)";

        g.appendChild(this.svgEl("line", {
          x1: s.x, y1: s.y, x2: e.x, y2: e.y,
          class: cls, "marker-end": marker
        }));
      }
    }

    // Nodes
    var nodeIds = Object.keys(graph.nodes);

    for (var ni = 0; ni < nodeIds.length; ni++) {
      var id = nodeIds[ni];
      var node = graph.nodes[id];
      var pos = positions[id];
      if (!pos) continue;

      var ng = this.svgEl("g", { id: "tg-" + id });
      var nw = pos.w, nh = pos.h;

      if (node.type === "data") {
        var fill = this._statusColor(node.status, "data");
        var drx = nw / 2, dry = nh / 2;
        ng.appendChild(this.svgEl("ellipse", {
          cx: pos.x, cy: pos.y, rx: drx, ry: dry,
          fill: fill, stroke: this._darken(fill, 0.25), "stroke-width": "1.5"
        }));
        var fontSize = (node.label && node.label.length > 4) ? "5" : "6.5";
        ng.appendChild(this.svgText(node.label || id, pos.x, pos.y + 0.5, {
          class: "tg-label", "font-size": fontSize
        }));
        ng.appendChild(this.svgText(node.status || "", pos.x, pos.y + dry + 9, {
          class: "tg-sublabel", "font-size": "5.5"
        }));
      } else if (node.type === "task" || node.type === "actor_method") {
        var fill2 = this._statusColor(node.status, "task");
        var rx = node.type === "actor_method" ? "5" : "3";

        if (node.type === "actor_method") {
          ng.appendChild(this.svgEl("rect", {
            x: pos.x - nw / 2 - 2, y: pos.y - nh / 2 - 2,
            width: nw + 4, height: nh + 4,
            fill: "none", stroke: this._darken(fill2, 0.35), "stroke-width": "1",
            rx: "7", ry: "7"
          }));
        }

        ng.appendChild(this.svgEl("rect", {
          x: pos.x - nw / 2, y: pos.y - nh / 2,
          width: nw, height: nh,
          fill: fill2, stroke: this._darken(fill2, 0.25), "stroke-width": "1.5",
          rx: rx, ry: rx
        }));
        ng.appendChild(this.svgText(node.label || id, pos.x, pos.y + 0.5, {
          class: "tg-label", "font-size": "6"
        }));
        ng.appendChild(this.svgText(node.status || "", pos.x, pos.y + nh / 2 + 9, {
          class: "tg-sublabel", "font-size": "5.5"
        }));
      }

      g.appendChild(ng);
    }

    svg.appendChild(g);
  }

  _ensureTaskGraphDefs(svg) {
    var defs = this.svgEl("defs");
    var markers = [
      { id: "tg-arrow-data", color: "#2196F3" },
      { id: "tg-arrow-control", color: "#FF9800" },
      { id: "tg-arrow-stateful", color: "#F44336" }
    ];
    for (var i = 0; i < markers.length; i++) {
      var m = markers[i];
      var marker = this.svgEl("marker", {
        id: m.id, markerWidth: "6", markerHeight: "5", refX: "5", refY: "2.5",
        orient: "auto", markerUnits: "strokeWidth"
      });
      marker.appendChild(this.svgEl("polygon", {
        points: "0 0, 6 2.5, 0 5", fill: m.color
      }));
      defs.appendChild(marker);
    }
    svg.insertBefore(defs, svg.firstChild);
  }

  _layoutTaskGraph(graph) {
    var nodes = graph.nodes;
    var edges = graph.edges || [];
    var TG_H_SPACING = RayDemo.TG_H_SPACING;
    var TG_V_SPACING = RayDemo.TG_V_SPACING;

    var incoming = {}, outgoing = {};
    var nodeIds = Object.keys(nodes);
    for (var i = 0; i < nodeIds.length; i++) {
      incoming[nodeIds[i]] = [];
      outgoing[nodeIds[i]] = [];
    }
    for (var j = 0; j < edges.length; j++) {
      var e = edges[j];
      if (outgoing[e.from]) outgoing[e.from].push(e.to);
      if (incoming[e.to])   incoming[e.to].push(e.from);
    }

    var levels = {};
    var visited = new Set();

    var computeLevel = function(id) {
      if (levels[id] !== undefined) return levels[id];
      if (visited.has(id)) return 0;
      visited.add(id);
      if (!incoming[id] || incoming[id].length === 0) { levels[id] = 0; return 0; }
      var maxP = 0;
      for (var k = 0; k < incoming[id].length; k++) {
        maxP = Math.max(maxP, computeLevel(incoming[id][k]));
      }
      levels[id] = maxP + 1;
      return levels[id];
    };

    for (var l = 0; l < nodeIds.length; l++) computeLevel(nodeIds[l]);

    var groups = {};
    var maxLvl = 0;
    var lvlKeys = Object.keys(levels);
    for (var m = 0; m < lvlKeys.length; m++) {
      var nid = lvlKeys[m];
      var lvl = levels[nid];
      if (!groups[lvl]) groups[lvl] = [];
      groups[lvl].push(nid);
      maxLvl = Math.max(maxLvl, lvl);
    }

    var positions = {};

    var MIN_TASK_W = 50, TEXT_PAD_X = 16;
    var TG_DATA_R = RayDemo.TG_DATA_R;
    var TG_TASK_H = RayDemo.TG_TASK_H;

    for (var lv = 0; lv <= maxLvl; lv++) {
      var grp = groups[lv] || [];
      // Sort: keep related nodes together by grouping by their primary incoming source
      grp.sort(function(a, b) {
        var ndA = nodes[a], ndB = nodes[b];
        // Tasks/actors first, then data
        var typeOrderA = ndA.type === "data" ? 1 : 0;
        var typeOrderB = ndB.type === "data" ? 1 : 0;
        if (typeOrderA !== typeOrderB) return typeOrderA - typeOrderB;
        // Group by primary incoming source to keep siblings together
        var srcA = incoming[a].length > 0 ? incoming[a][0] : a;
        var srcB = incoming[b].length > 0 ? incoming[b][0] : b;
        if (srcA !== srcB) return srcA.localeCompare(srcB);
        return a.localeCompare(b);
      });

      var nodeWidths = [];
      var totalNodesW = 0;
      for (var gi = 0; gi < grp.length; gi++) {
        var nid = grp[gi];
        var nd = nodes[nid];
        var nw;
        if (nd.type === "data") {
          var dl = nd.label || nid;
          var dataW = this._measureTextWidth(dl) + 12;
          nw = Math.max(TG_DATA_R * 2, dataW);
        } else {
          var label = nd.label || nid;
          var textW = this._measureTextWidth(label);
          nw = Math.max(MIN_TASK_W, textW + TEXT_PAD_X);
        }
        nodeWidths.push(nw);
        totalNodesW += nw;
      }

      var totalGap = (grp.length - 1) * TG_H_SPACING;
      var totalW = totalNodesW + totalGap;
      var x = -totalW / 2;
      for (var gi = 0; gi < grp.length; gi++) {
        var nw = nodeWidths[gi];
        var nid = grp[gi];
        var nd = nodes[nid];
        var nh = nd.type === "data" ? TG_DATA_R * 2 : TG_TASK_H;
        positions[nid] = { x: x + nw / 2, y: lv * TG_V_SPACING, w: nw, h: nh };
        x += nw + TG_H_SPACING;
      }
    }

    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    var posKeys = Object.keys(positions);
    for (var pi = 0; pi < posKeys.length; pi++) {
      var pp = positions[posKeys[pi]];
      var halfW = pp.w / 2;
      var halfH = pp.h / 2;
      minX = Math.min(minX, pp.x - halfW);
      maxX = Math.max(maxX, pp.x + halfW);
      minY = Math.min(minY, pp.y - halfH);
      maxY = Math.max(maxY, pp.y + halfH);
    }

    return {
      positions: positions,
      width: maxX - minX,
      height: maxY - minY,
      offsetX: -minX,
      offsetY: -minY
    };
  }

  _tgEdgePoint(pos, node, target) {
    if (!node) return pos;
    var cx = pos.x, cy = pos.y;
    var dx = target.x - cx;
    var dy = target.y - cy;

    if (dx === 0 && dy === 0) return pos;

    if (node.type === "data") {
      var erx = (pos.w || RayDemo.TG_DATA_R * 2) / 2 + 2;
      var ery = (pos.h || RayDemo.TG_DATA_R * 2) / 2 + 2;
      if (erx === ery) {
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        return { x: cx + (dx / dist) * erx, y: cy + (dy / dist) * erx };
      }
      var d = Math.sqrt((dx / erx) * (dx / erx) + (dy / ery) * (dy / ery)) || 1;
      var t = 1 / d;
      return { x: cx + dx * t, y: cy + dy * t };
    }

    // Rectangle edge intersection
    var hw = (pos.w || RayDemo.TG_TASK_W) / 2 + 2;
    var hh = (pos.h || RayDemo.TG_TASK_H) / 2 + 2;
    var ex, ey;
    if (Math.abs(dx) * hh > Math.abs(dy) * hw) {
      ex = cx + (dx > 0 ? hw : -hw);
      ey = cy + dy * (ex - cx) / dx;
    } else {
      ey = cy + (dy > 0 ? hh : -hh);
      ex = cx + dx * (ey - cy) / dy;
    }
    return { x: ex, y: ey };
  }

  _statusColor(status, type) {
    switch (status) {
      case "completed": return type === "data" ? "#2196F3" : "#4CAF50";
      case "running":   return "#FF9800";
      case "pending":   return "#bdbdbd";
      case "failed":    return "#F44336";
      default:          return "#bdbdbd";
    }
  }

  _darken(hex, amt) {
    if (!hex || !hex.startsWith("#")) return hex;
    var num = parseInt(hex.slice(1), 16);
    var r = Math.max(0, Math.round(((num >> 16) & 255) * (1 - amt)));
    var g = Math.max(0, Math.round(((num >> 8) & 255) * (1 - amt)));
    var b = Math.max(0, Math.round((num & 255) * (1 - amt)));
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  /* ================================================================
     EVENT LOG
     ================================================================ */

  clearEventLog() {
    document.getElementById("event-log").innerHTML =
      '<div class="log-empty">No events yet</div>';
  }

  renderEventLog(currentStep) {
    var container = document.getElementById("event-log");
    container.innerHTML = "";

    var steps = [];
    for (var s = 1; s <= this.maxStepSeen; s++) {
      if (this.eventLogMap[s]) steps.push({ step: s, event: this.eventLogMap[s] });
    }

    if (steps.length === 0) {
      container.innerHTML = '<div class="log-empty">No events yet</div>';
      return;
    }

    // Newest on top
    for (var i = steps.length - 1; i >= 0; i--) {
      var stepData = steps[i];
      var div = document.createElement("div");
      div.className = "log-entry" + (stepData.step === currentStep ? " current" : "");
      div.dataset.step = stepData.step;

      var self = this;
      (function(st) {
        div.addEventListener("click", function() { self.gotoStep(st); });
      })(stepData.step);

      var stepSpan = document.createElement("span");
      stepSpan.className = "log-step";
      stepSpan.textContent = "[" + stepData.step + "]";

      var phaseSpan = document.createElement("span");
      phaseSpan.className = "log-phase";
      phaseSpan.textContent = (stepData.event.phase || "").replace(/_/g, " ");

      var textSpan = document.createElement("span");
      textSpan.className = "log-text";
      textSpan.textContent = stepData.event.description || "";

      div.appendChild(stepSpan);
      div.appendChild(phaseSpan);
      div.appendChild(textSpan);
      container.appendChild(div);
    }

    var cur = container.querySelector(".log-entry.current");
    if (cur) cur.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

/* ================================================================
   BOOT
   ================================================================ */

document.addEventListener("DOMContentLoaded", function() {
  window.rayDemo = new RayDemo();
  window.rayDemo.fetchPrograms();
});

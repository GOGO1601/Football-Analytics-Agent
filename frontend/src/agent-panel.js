/**
 * agent-panel.js
 * Floating "Ask the analyst" panel on every dashboard page. Sends questions through
 * AgentApi (src/agent-api.js) to the FastAPI backend and shows the answer, its grounding
 * status, evidence and data warnings. It never computes statistics and never changes the
 * charts: chart_actions are listed as suggestions only.
 *
 * All backend text is inserted with textContent (via el()); innerHTML is never used.
 * window.AgentPanel exposes open/close/renderResult/renderError for tests.
 */
(function () {
    "use strict";

    if (!window.AgentApi) {
        console.warn("[agent] agent-api.js is not loaded; the analyst panel is disabled.");
        return;
    }
    const Api = window.AgentApi;

    // Questions the current backend (including offline-mock mode) genuinely answers.
    const EXAMPLES = [
        "Was home advantage lower during COVID in Serie A?",
        "Do teams that take more shots get more points?",
        "Do yellow cards cost teams points?",
        "Do red cards matter in La Liga?",
        "Does scoring more goals in the Bundesliga lead to more points?",
    ];

    const OUTCOME_INFO = {
        verified:    { badge: "✓ Verified",   note: "Every figure in this answer was checked against the tool results." },
        no_evidence: { badge: "No evidence",  note: "The available data could not answer this question, so no statistics are given." },
        fallback:    { badge: "Fallback",     note: "The written answer failed the fact check, so the analysis tools' own summaries are shown instead." },
        unverified:  { badge: "⚠ Unverified", note: "This answer could not be verified against the tool results. Treat it with caution." },
    };

    const PAGES = { q1: ["Trends (Q1)", "index.html"], q2: ["Attack (Q2)", "q2-offensive-vs-success.html"],
        q3: ["Discipline (Q3)", "q3-disciplinary.html"] };
    const STATUS_TEXT = { unknown: "Checking…", ready: "Online", not_ready: "Not ready", offline: "Backend offline" };
    const STATUS_MIN_INTERVAL_MS = 5000;

    // ------------------------------------------------------------------ DOM helper (text only)
    function el(tag, attrs, ...children) {
        const node = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs || {})) {
            if (v == null || v === false) continue;
            if (k === "class") node.className = v;
            else if (k === "text") node.textContent = String(v);
            else node.setAttribute(k, v === true ? "" : String(v));
        }
        for (const c of children.flat()) {
            if (c == null || c === false) continue;
            node.append(c instanceof Node ? c : document.createTextNode(String(c)));
        }
        return node;
    }

    const arr = v => (Array.isArray(v) ? v : []);
    const str = v => (typeof v === "string" ? v : v == null ? "" : String(v));

    // ------------------------------------------------------------------ result rendering
    function section(title, ...content) {
        return el("section", { class: "agent-section" }, el("h3", { class: "agent-section-title", text: title }), ...content);
    }

    function describeAction(a) {
        if (!a || !PAGES[a.page]) return null;
        const parts = [PAGES[a.page][0]];
        if (a.metric) parts.push(str(a.metric));
        if (a.x_metric) parts.push(`x = ${str(a.x_metric)}`);
        if (a.league) parts.push(a.league === "all" ? "all leagues" : str(a.league));
        if (Array.isArray(a.teams) && a.teams.length) parts.push(a.teams.map(str).join(", "));
        if (a.season_start || a.season_end) parts.push(`${str(a.season_start || "…")} to ${str(a.season_end || "…")}`);
        return { text: parts.join(" · "), href: PAGES[a.page][1] };
    }

    function detailsBlock(d) {
        const g = d.grounding || {};
        const rows = [];
        const add = (k, v) => { if (v !== "" && v != null) rows.push(el("dt", { text: k }), el("dd", { text: v })); };
        add("Question type", str(d.intent));
        add("Analysts", arr(d.analysts).map(str).join(", "));
        add("Grounding", `${g.passed ? "passed" : "failed"} · ${Number(g.checked) || 0} claims checked` +
            (arr(g.unsupported).length ? ` · unsupported: ${arr(g.unsupported).map(str).join(", ")}` : ""));
        add("Retries", str(d.retries ?? ""));
        if (typeof d.duration_ms === "number") add("Time", `${(d.duration_ms / 1000).toFixed(1)} s`);
        const u = d.usage || {};
        if (arr(u.models).length) add("Language model", `${arr(u.models).map(str).join(", ")} · ${Number(u.llm_calls) || 0} calls`);
        add("Request id", str(d.request_id));

        const parts = [el("dl", { class: "agent-dl" }, rows)];
        const issues = arr(g.issues);
        if (issues.length) {
            parts.push(el("h4", { text: "Grounding issues" }),
                el("ul", { class: "agent-list" }, issues.map(i => el("li", { text: `${str(i && i.kind)}: ${str(i && i.claim)} — ${str(i && i.detail)}` }))));
        }
        const errors = arr(d.errors);
        if (errors.length) {
            parts.push(el("h4", { text: "Problems along the way" }),
                el("ul", { class: "agent-list" }, errors.map(e => el("li", { text: `${str(e && e.node)} · ${str(e && e.kind)}: ${str(e && e.message)}` }))));
        }
        const tools = arr(d.tool_results);
        if (tools.length) {
            parts.push(el("h4", { text: "Tool calls" }), el("ul", { class: "agent-list agent-mono" }, tools.map(t => {
                const status = t && t.ok ? "ok" : `failed: ${str(t && t.error && t.error.message)}`;
                return el("li", { text: `${str(t && t.call_id)} ${str(t && t.tool)}(${JSON.stringify((t && t.args) || {})}) — ${status}` });
            })));
            const raw = el("pre", { class: "agent-raw", text: JSON.stringify(tools.map(t => t && t.result), null, 2) });
            parts.push(el("details", { class: "agent-raw-wrap" }, el("summary", { text: "Raw tool output" }), raw));
        }
        const trace = arr(d.trace);
        if (trace.length) {
            parts.push(el("h4", { text: "Workflow steps" }),
                el("p", { class: "agent-mono", text: trace.map(t => `${str(t && t.node)} ${Number(t && t.ms || 0).toFixed(0)} ms` +
                        (t && t.status === "error" ? " (error)" : "")).join(" → ") }));
        }
        return el("details", { class: "agent-details" }, el("summary", { text: "Analysis details" }), parts);
    }

    function renderResult(container, d) {
        const outcome = Api.OUTCOMES.includes(d.outcome) ? d.outcome : "unverified";
        const info = OUTCOME_INFO[outcome];
        const card = el("article", { class: `agent-result agent-result--${outcome}`, "data-outcome": outcome });

        card.append(el("div", { class: "agent-outcome" },
            el("span", { class: `agent-badge agent-badge--${outcome}`, text: info.badge }),
            el("span", { class: "agent-outcome-note", text: info.note })));
        card.append(el("p", { class: "agent-answer", text: str(d.answer) }));

        const stats = arr(d.key_stats).filter(s => s && s.value != null);
        if (stats.length) {
            card.append(el("ul", { class: "agent-stats" }, stats.map(s =>
                el("li", { class: "agent-stat", title: s.source ? `from ${str(s.source)}` : null },
                    el("span", { class: "agent-stat-value", text: str(s.value) }),
                    el("span", { class: "agent-stat-label", text: str(s.label) })))));
        }
        const evidence = arr(d.evidence).map(str).filter(Boolean);
        if (evidence.length && !(evidence.length === 1 && evidence[0] === str(d.answer))) {
            card.append(section("Evidence", el("ul", { class: "agent-list" }, evidence.map(e => el("li", { text: e })))));
        }
        if (str(d.explanation)) card.append(el("p", { class: "agent-explanation", text: str(d.explanation) }));

        const warnings = arr(d.warnings).map(str).filter(Boolean);
        if (warnings.length) {
            card.append(el("section", { class: "agent-section agent-warnings", role: "note" },
                el("h3", { class: "agent-section-title", text: `Data warnings (${warnings.length})` }),
                el("ul", { class: "agent-list" }, warnings.map(w => el("li", { text: w })))));
        }
        const actions = arr(d.chart_actions).map(describeAction).filter(Boolean);
        if (actions.length) {
            card.append(section("Suggested chart views",
                el("p", { class: "agent-muted", text: "Suggestions only — open the page and set these options yourself." }),
                el("ul", { class: "agent-list" }, actions.map(a => el("li", {}, el("a", { href: a.href, text: a.text }))))));
        }
        card.append(detailsBlock(d));
        container.replaceChildren(card);
        return card;
    }

    const ERROR_TITLES = {
        validation: "Check your question", busy: "Still working", network: "Backend not reachable",
        timeout: "Took too long", cancelled: "Cancelled", http: "The service reported a problem",
        parse: "Unreadable response", contract: "Result not shown",
    };

    function renderError(container, err, onRetry, onCheck) {
        const kind = (err && err.kind) || "parse";
        const card = el("div", { class: `agent-error agent-error--${kind}`, role: "alert", "data-kind": kind },
            el("strong", { text: ERROR_TITLES[kind] || "Something went wrong" }),
            el("p", { text: str(err && err.message) || "The request failed." }));
        if (kind === "network") {
            card.append(el("p", { class: "agent-muted" }, "Start it from agent-backend with: ",
                el("code", { text: "python -m uvicorn app.main:create_app --factory --port 8000" })));
        }
        if (err && err.requestId && !str(err.message).includes(err.requestId)) {
            card.append(el("p", { class: "agent-muted agent-mono", text: `Request id: ${str(err.requestId)}` }));
        }
        const buttons = el("div", { class: "agent-error-actions" });
        if (onRetry && kind !== "validation") buttons.append(el("button", { type: "button", class: "agent-btn agent-btn--ghost", text: "Try again" }));
        if (onCheck && (kind === "network" || kind === "http" || kind === "timeout")) {
            buttons.append(el("button", { type: "button", class: "agent-btn agent-btn--ghost", text: "Check service" }));
        }
        [...buttons.children].forEach(b => b.addEventListener("click", b.textContent === "Try again" ? onRetry : onCheck));
        if (buttons.children.length) card.append(buttons);
        container.replaceChildren(card);
        return card;
    }

    // ------------------------------------------------------------------ panel
    function build() {
        const launcher = el("button", { type: "button", class: "agent-launcher", id: "agentLauncher",
                "aria-expanded": "false", "aria-controls": "agentPanel" },
            el("span", { class: "agent-dot", "data-state": "unknown", "aria-hidden": "true" }),
            el("span", { text: "Ask the analyst" }));

        const textarea = el("textarea", { id: "agentQuestion", class: "agent-input", rows: "3",
            placeholder: "e.g. Do teams that take more shots get more points?",
            "aria-describedby": "agentCount agentFormError" });
        const submit = el("button", { type: "submit", id: "agentSubmit", class: "agent-btn", text: "Ask" });
        const cancel = el("button", { type: "button", id: "agentCancel", class: "agent-btn agent-btn--ghost", text: "Cancel", hidden: true });
        const count = el("span", { id: "agentCount", class: "agent-count", text: `0 / ${Api.MAX_QUESTION_CHARS}` });
        const formError = el("p", { id: "agentFormError", class: "agent-form-error", role: "alert" });
        const form = el("form", { id: "agentForm", class: "agent-form", novalidate: true },
            el("label", { for: "agentQuestion", class: "agent-label", text: "Your question" }),
            textarea, el("div", { class: "agent-form-row" }, count, el("span", { class: "agent-spacer" }), cancel, submit), formError);

        const statusPill = el("span", { id: "agentStatus", class: "agent-status", "data-state": "unknown", text: STATUS_TEXT.unknown });
        const closeBtn = el("button", { type: "button", class: "agent-close", "aria-label": "Close the analyst panel", text: "×" });
        const examples = el("div", { class: "agent-examples" },
            el("span", { class: "agent-muted", text: "Try one:" }),
            EXAMPLES.map(q => el("button", { type: "button", class: "agent-chip", text: q })));
        const statusLine = el("p", { id: "agentProgress", class: "agent-progress", "aria-live": "polite" });
        const result = el("div", { id: "agentResult", class: "agent-result-area", "aria-live": "polite" });

        const panel = el("aside", { id: "agentPanel", class: "agent-panel", role: "dialog", "aria-modal": "false",
                "aria-labelledby": "agentPanelTitle", hidden: true },
            el("header", { class: "agent-header" },
                el("h2", { id: "agentPanelTitle", class: "agent-title", text: "Ask the analyst" }), statusPill, closeBtn),
            el("div", { class: "agent-body", id: "agentBody" },
                el("p", { class: "agent-intro", text: "Answers come from statistics tools run on Premier League, La Liga, Bundesliga, " +
                        "Serie A and Ligue 1 matches (1993-94 to 2025-26), then fact-checked. Not every question can be answered, " +
                        "and the panel says when an answer could not be verified." }),
                examples, form, statusLine, result));

        document.body.append(launcher, panel);
        return { launcher, panel, body: panel.querySelector("#agentBody"), textarea, submit, cancel, count, formError, statusPill, closeBtn, examples, statusLine, result };
    }

    function init() {
        if (document.getElementById("agentPanel")) return;
        const ui = build();
        let busy = false, cancelCtl = null, tick = null, lastQuestion = "", lastStatusCheck = 0;

        function setStatus(state, llm) {
            ui.statusPill.dataset.state = state;
            ui.statusPill.textContent = STATUS_TEXT[state] + (state === "ready" && llm ? ` · ${llm}` : "");
            ui.launcher.querySelector(".agent-dot").dataset.state = state;
        }

        async function checkStatus() {
            const now = Date.now();
            if (now - lastStatusCheck < STATUS_MIN_INTERVAL_MS) return;
            lastStatusCheck = now;
            setStatus("unknown");
            const r = await Api.ready();
            setStatus(r.state, r.llm);
        }

        function scrollToResult() {
            const top = ui.result.getBoundingClientRect().top - ui.body.getBoundingClientRect().top + ui.body.scrollTop - 8;
            const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
            ui.body.scrollTo({ top, behavior: smooth ? "smooth" : "auto" });
        }

        function updateCount() {
            const n = Api.normalise(ui.textarea.value).length;
            ui.count.textContent = `${n} / ${Api.MAX_QUESTION_CHARS}`;
            ui.count.classList.toggle("agent-count--over", n > Api.MAX_QUESTION_CHARS);
        }

        function setBusy(on) {
            busy = on;
            ui.submit.disabled = on;
            ui.submit.textContent = on ? "Asking…" : "Ask";
            ui.textarea.readOnly = on;
            ui.cancel.hidden = !on;
            ui.examples.querySelectorAll("button").forEach(b => { b.disabled = on; });
            ui.panel.setAttribute("aria-busy", on ? "true" : "false");
            clearInterval(tick);
            if (on) {
                const t0 = Date.now();
                ui.statusLine.textContent = "Analysing…";
                tick = setInterval(() => { ui.statusLine.textContent = `Analysing… ${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
            }
        }

        async function submitQuestion(text) {
            if (busy) return;                                     // duplicate submissions are ignored
            ui.formError.textContent = "";
            const v = Api.validateQuestion(text);
            if (!v.ok) { ui.formError.textContent = v.message; ui.textarea.focus(); return; }
            lastQuestion = v.question;
            cancelCtl = new AbortController();
            setBusy(true);
            const t0 = performance.now();
            try {
                const data = await Api.ask(v.question, { signal: cancelCtl.signal });
                renderResult(ui.result, data);
                scrollToResult();
                ui.statusLine.textContent = `Answered in ${((performance.now() - t0) / 1000).toFixed(1)} s. Ask another question any time.`;
                setStatus("ready", arr(data.usage && data.usage.models)[0]);
            } catch (err) {
                ui.statusLine.textContent = "";
                if (err && err.kind === "validation") { ui.formError.textContent = err.message; }
                else { renderError(ui.result, err, () => submitQuestion(lastQuestion), () => { lastStatusCheck = 0; checkStatus(); }); scrollToResult(); }
                if (err && err.kind === "network") setStatus("offline");
            } finally {
                setBusy(false);
            }
        }

        function open() {
            ui.panel.hidden = false;
            ui.launcher.setAttribute("aria-expanded", "true");
            ui.launcher.classList.add("agent-launcher--open");
            ui.textarea.focus();
            checkStatus();
        }
        function close() {
            ui.panel.hidden = true;
            ui.launcher.setAttribute("aria-expanded", "false");
            ui.launcher.classList.remove("agent-launcher--open");
            ui.launcher.focus();
        }

        ui.launcher.addEventListener("click", () => (ui.panel.hidden ? open() : close()));
        ui.closeBtn.addEventListener("click", close);
        ui.panel.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
        ui.textarea.addEventListener("input", updateCount);
        ui.textarea.addEventListener("keydown", e => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submitQuestion(ui.textarea.value); }
        });
        ui.panel.querySelector("form").addEventListener("submit", e => { e.preventDefault(); submitQuestion(ui.textarea.value); });
        ui.cancel.addEventListener("click", () => cancelCtl && cancelCtl.abort());
        ui.examples.querySelectorAll(".agent-chip").forEach(chip => chip.addEventListener("click", () => {
            ui.textarea.value = chip.textContent;
            updateCount();
            submitQuestion(chip.textContent);
        }));

        window.AgentPanel = { open, close, renderResult, renderError, submit: submitQuestion, isBusy: () => busy };
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
/**
 * agent-api.js
 * Small client for the FastAPI agent backend (agent-backend/app/main.py). No dependencies.
 *
 *   AgentApi.ask(question, {timeoutMs, signal})  -> Promise<AskResponse>   (throws AgentApiError)
 *   AgentApi.ready({timeoutMs})                   -> Promise<{state, llm, matches}>
 *   AgentApi.validateQuestion(text)               -> {ok, question} | {ok:false, message}
 *   AgentApi.checkResponse(json)                  -> list of contract problems ([] = usable)
 *
 * Base URL: window.AGENT_API_BASE if set before this script, else http://127.0.0.1:8000.
 * The client never logs the question or the response body; failures log kind/status/request id.
 * The backend stays the authority on validation and grounding; the checks here are defensive.
 */
(function () {
    "use strict";

    const DEFAULT_BASE = "http://127.0.0.1:8000";
    const MAX_QUESTION_CHARS = 1000;             // same limit as agents/state.py MAX_QUESTION_CHARS
    const DEFAULT_TIMEOUT_MS = 60000;
    const OUTCOMES = ["verified", "no_evidence", "fallback", "unverified"];
    // outcome -> what grounding.passed must be (mirrors app/schemas.py EXPECTED_PASSED)
    const EXPECTED_PASSED = { verified: true, no_evidence: true, fallback: true, unverified: false };

    class AgentApiError extends Error {
        /** kind: validation | busy | network | timeout | cancelled | http | parse | contract */
        constructor(kind, message, extra) {
            super(message);
            this.name = "AgentApiError";
            this.kind = kind;
            Object.assign(this, { status: null, requestId: null, fieldErrors: [] }, extra || {});
        }
    }

    function baseUrl() {
        const b = (typeof window.AGENT_API_BASE === "string" && window.AGENT_API_BASE.trim()) || DEFAULT_BASE;
        return b.replace(/\/+$/, "");
    }

    function normalise(text) {
        return String(text == null ? "" : text).split(/\s+/).filter(Boolean).join(" ");
    }

    function validateQuestion(text) {
        const question = normalise(text);
        if (!question) return { ok: false, message: "Please type a question first." };
        if (question.length > MAX_QUESTION_CHARS) {
            return { ok: false, message: `Questions can be at most ${MAX_QUESTION_CHARS} characters ` +
                    `(this one has ${question.length}).` };
        }
        return { ok: true, question };
    }

    function isObj(v) { return v !== null && typeof v === "object" && !Array.isArray(v); }

    /** Contract checks before anything is shown. Optional fields may be missing; core ones may not. */
    function checkResponse(d) {
        const problems = [];
        if (!isObj(d)) return ["response is not a JSON object"];
        if (!OUTCOMES.includes(d.outcome)) problems.push("unknown outcome");
        if (typeof d.answer !== "string" || !d.answer.trim()) problems.push("missing answer");
        if (!isObj(d.grounding) || typeof d.grounding.passed !== "boolean") problems.push("missing grounding status");
        if (typeof d.verified !== "boolean") problems.push("missing verified flag");
        if (!problems.length) {
            if (d.verified !== (d.outcome === "verified")) problems.push("verified flag contradicts outcome");
            if (d.grounding.passed !== EXPECTED_PASSED[d.outcome]) problems.push("grounding status contradicts outcome");
        }
        return problems;
    }

    function logFailure(err) {
        // never the question or the body: enough to find the request in the server log
        console.warn("[agent] request failed", { kind: err.kind, status: err.status, requestId: err.requestId });
    }

    async function readJson(res) {
        const text = await res.text();
        try { return JSON.parse(text); } catch (_) { return undefined; }
    }

    function httpError(res, body) {
        const requestId = (isObj(body) && body.request_id) || res.headers.get("X-Request-ID");
        const fieldErrors = isObj(body) && Array.isArray(body.errors)
            ? body.errors.map(e => String((e && e.msg) || "").replace(/^Value error,\s*/, "")).filter(Boolean)
            : [];
        let message;
        if (res.status === 422) {
            message = fieldErrors.length ? `The question was not accepted: ${fieldErrors[0]}`
                : "The question was not accepted by the service.";
        } else if (res.status === 413) {
            message = "The question is too long to send.";
        } else if (res.status >= 500) {
            message = "The analysis service hit an internal error. Please try again; if it keeps happening, " +
                "check the backend log" + (requestId ? ` (request id ${requestId}).` : ".");
        } else {
            message = `The analysis service answered with HTTP ${res.status}.`;
        }
        return new AgentApiError("http", message, { status: res.status, requestId, fieldErrors });
    }

    /** One request at a time per page: a second ask() while one is pending is refused. */
    let pending = null;

    async function ask(text, opts) {
        const options = opts || {};
        const v = validateQuestion(text);
        if (!v.ok) throw new AgentApiError("validation", v.message);
        if (pending) throw new AgentApiError("busy", "Please wait for the current question to finish.");

        const controller = new AbortController();
        const timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;
        let timedOut = false;
        const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
        const onCancel = () => controller.abort();
        if (options.signal) options.signal.addEventListener("abort", onCancel, { once: true });
        const url = `${baseUrl()}/api/ask`;
        pending = controller;

        try {
            let res;
            try {
                res = await fetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json", "Accept": "application/json" },
                    body: JSON.stringify({ question: v.question }),
                    signal: controller.signal,
                });
            } catch (e) {
                if (timedOut) throw new AgentApiError("timeout",
                    `The analysis took longer than ${Math.round(timeoutMs / 1000)} s and was stopped. Please try again.`);
                if (controller.signal.aborted) throw new AgentApiError("cancelled", "The question was cancelled.");
                throw new AgentApiError("network",
                    `Can't reach the analysis service at ${baseUrl()}. Make sure the backend is running ` +
                    "and that this page's address is allowed in CORS_ORIGINS.");
            }
            const body = await readJson(res);
            if (!res.ok) throw httpError(res, body);
            const requestId = (isObj(body) && body.request_id) || res.headers.get("X-Request-ID");
            if (body === undefined) {
                throw new AgentApiError("parse", "The analysis service sent a response that could not be read.",
                    { status: res.status, requestId });
            }
            const problems = checkResponse(body);
            if (problems.length) {
                throw new AgentApiError("contract",
                    "The analysis service returned an incomplete or inconsistent result, so it is not shown.",
                    { status: res.status, requestId, problems });
            }
            return body;
        } catch (err) {
            if (err instanceof AgentApiError) {
                if (err.kind !== "cancelled") logFailure(err);
                throw err;
            }
            const wrapped = new AgentApiError("parse", "The analysis service response could not be processed.");
            logFailure(wrapped);
            throw wrapped;
        } finally {
            clearTimeout(timer);
            if (options.signal) options.signal.removeEventListener("abort", onCancel);
            pending = null;
        }
    }

    /** /health/ready: never calls the LLM or a tool on the backend. */
    async function ready(opts) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), (opts && opts.timeoutMs) || 4000);
        try {
            const res = await fetch(`${baseUrl()}/health/ready`, { signal: controller.signal });
            const body = await readJson(res);
            const checks = isObj(body) && isObj(body.checks) ? body.checks : {};
            const llm = typeof checks.llm === "string" ? checks.llm.split(" ")[0] : null;
            if (res.ok && isObj(body) && body.status === "ready") return { state: "ready", llm, dataset: checks.dataset || null };
            return { state: "not_ready", llm, dataset: checks.dataset || null };
        } catch (_) {
            return { state: "offline", llm: null, dataset: null };
        } finally {
            clearTimeout(timer);
        }
    }

    window.AgentApi = {
        ask, ready, validateQuestion, checkResponse, normalise, baseUrl,
        isBusy: () => pending !== null,
        AgentApiError, MAX_QUESTION_CHARS, OUTCOMES,
    };
})();
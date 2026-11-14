"""
Stage 9: the dashboard's "Ask the analyst" panel (frontend/src/agent-api.js + agent-panel.js),
driven in headless Chromium with Playwright (already in the dev extra).

  * mocked tests: the API is routed to a fake origin (http://agent.test) and answered with REAL
    AskResponse JSON produced by the backend in-process (app.main.create_app + TestClient)
  * integration tests: a real uvicorn server (offline-mock LLM) on a free port
  * the pages are served from the repo root by a local static server, as WebStorm does

Needs Chromium for Playwright:  python -m playwright install chromium
If it is missing, every test here is SKIPPED with that instruction (they are not silently passed).
d3 is loaded from the CDN like the real pages; set D3_LOCAL_PATH=/path/to/d3.min.js to serve a local copy.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import socket
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents import demo
from agents import graph as agent_graph
from agents import guard as G
from app.config import Settings
from app.main import create_app
from llm.mock import MockLLM

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
FAKE = "http://agent.test"
SHOTS_Q = "Do teams that take more shots get more points?"
COVID_Q = "Was home advantage lower during COVID in Serie A?"
XSS = '<img src=x onerror="window.__pwned=1"><script>window.__pwned=2</script>'

class _StaticServer(ThreadingHTTPServer):
    # socketserver's default listen backlog is 5. On Windows, connections beyond the backlog are
    # refused at once (Linux queues them), so a page that requests several scripts in parallel can
    # randomly lose one, e.g. agent-api.js, and the panel then disables itself.
    request_queue_size = 128
    daemon_threads = True

# ---------------------------------------------------------------------------
# Infrastructure: static server, browser, real backend
# ---------------------------------------------------------------------------
class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def site():
    server = _StaticServer(("127.0.0.1", 0), partial(_Quiet, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium for Playwright is not available ({type(e).__name__}). "
                        "Run: python -m playwright install chromium")
        yield b
        b.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def backend(cors_origins: str):
    import uvicorn
    port = _free_port()
    app = create_app(Settings(_env_file=None, cors_origins=cors_origins, preload_data=True))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 30
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(10)


def _new_page(browser, site, path="frontend/home.html", api_base=FAKE):
    ctx = browser.new_context()
    d3_local = os.environ.get("D3_LOCAL_PATH")
    if d3_local:
        body = Path(d3_local).read_bytes()
        ctx.route("**/d3.v7.min.js", lambda r: r.fulfill(body=body, content_type="application/javascript"))
    ctx.route("https://fonts.googleapis.com/**", lambda r: r.abort())
    ctx.route("https://fonts.gstatic.com/**", lambda r: r.abort())
    if api_base:
        ctx.add_init_script(f"window.AGENT_API_BASE = {json.dumps(api_base)};")
    page = ctx.new_page()
    page.errors, page.failed = [], []
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    page.on("requestfailed", lambda r: page.failed.append(f"{r.url} ({r.failure})"))
    page.on("response", lambda r: r.status >= 400 and page.failed.append(f"{r.url} (HTTP {r.status})"))
    page.goto(f"{site}/{path}")
    try:
        page.wait_for_selector("#agentLauncher", timeout=15000)
    except sync_api.TimeoutError:
        scripts = page.evaluate("[...document.scripts].map(s => s.getAttribute('src')).filter(Boolean)")
        missing = [f for f in ("src/agent-api.js", "src/agent-panel.js") if f not in scripts]
        api_loaded = page.evaluate("typeof window.AgentApi === 'object'")
        failed = [u for u in page.failed if "/frontend/" in u]
        ctx.close()
        pytest.fail(f"{path}: the analyst panel did not appear. Missing <script> tags: {missing or 'none'}; "
                    f"AgentApi loaded: {api_loaded}; failed requests: {failed or 'none'}; "
                    f"page errors: {page.errors or 'none'}", pytrace=False)
    return page


@pytest.fixture
def page(browser, site):
    p = _new_page(browser, site)
    yield p
    p.context.close()


# ---------------------------------------------------------------------------
# Real backend responses (generated in-process, no server, no LLM)
# ---------------------------------------------------------------------------
def _post(app, body) -> tuple[int, dict]:
    r = TestClient(app).post("/api/ask", json=body)
    return r.status_code, r.json()


def _settings():
    return Settings(_env_file=None, preload_data=False)


@pytest.fixture(scope="module")
def responses() -> dict:
    out = {}
    out["verified"] = _post(create_app(_settings()), {"question": SHOTS_Q})[1]
    out["covid"] = _post(create_app(_settings()), {"question": COVID_Q})[1]
    for name in ("no_evidence", "fallback", "retry"):
        q, script = demo.SCENARIOS[name]
        out[name] = _post(create_app(_settings(), graph=agent_graph.build_graph(MockLLM(script))), {"question": q})[1]
    with pytest.MonkeyPatch.context() as mp:
        real = G.check
        mp.setattr(G, "check", lambda s: {**real(s), "passed": False,
                                          "issues": [{"kind": "unsupported_number", "field": "answer",
                                                      "claim": "0.9", "detail": "not in the evidence"}]})
        out["unverified"] = _post(create_app(_settings()), {"question": SHOTS_Q})[1]
    out["422"] = _post(create_app(_settings()), {"question": 42})[1]

    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("GROQ_API_KEY=gsk_secret")
    c = TestClient(create_app(_settings(), graph=Boom()), raise_server_exceptions=False)
    r = c.post("/api/ask", json={"question": SHOTS_Q})
    out["500"] = r.json()
    assert [out[k]["outcome"] for k in ("verified", "no_evidence", "fallback", "unverified")] == \
           ["verified", "no_evidence", "fallback", "unverified"]
    return out


def route_api(page, body, status=200, path="/api/ask", log=None):
    def handle(route):
        if log is not None:
            log.append(route.request)
        route.fulfill(status=status, content_type="application/json",
                      body=body if isinstance(body, str) else json.dumps(body))
    page.route(f"{FAKE}{path}", handle)


def hold_api(page, held: list):
    page.route(f"{FAKE}/api/ask", lambda r: held.append(r))


def release(held: list):
    """Abort requests a test left hanging, so the browser context closes cleanly."""
    for r in held:
        with contextlib.suppress(Exception):
            r.abort()


def call_ask(page, question, **opts):
    return page.evaluate("""async ([q, opts]) => {
        try { return {ok: true, data: await AgentApi.ask(q, opts)}; }
        catch (e) { return {ok: false, kind: e.kind, status: e.status, message: e.message, requestId: e.requestId}; }
    }""", [question, opts])


def ui_ask(page, question):
    if page.locator("#agentPanel").is_hidden():
        page.click("#agentLauncher")
    page.fill("#agentQuestion", question)
    page.click("#agentSubmit")
    page.wait_for_selector("#agentResult .agent-result, #agentResult .agent-error, #agentFormError:not(:empty)")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
def test_default_base_url_endpoint_and_request_shape(browser, site, responses):
    p = _new_page(browser, site, api_base=None)                    # no override: the documented default
    try:
        seen = []
        p.route("http://127.0.0.1:8000/api/ask", lambda r: (seen.append(r.request), r.fulfill(
            status=200, content_type="application/json", body=json.dumps(responses["verified"]))))
        assert p.evaluate("AgentApi.baseUrl()") == "http://127.0.0.1:8000"
        res = call_ask(p, "  Do teams that take   more shots get more points? ")
        assert res["ok"] and len(seen) == 1
        req = seen[0]
        assert req.method == "POST" and req.url == "http://127.0.0.1:8000/api/ask"
        assert req.headers["content-type"] == "application/json"
        assert req.post_data_json == {"question": SHOTS_Q}             # normalised like the backend
    finally:
        p.context.close()


def test_successful_response_is_parsed(page, responses):
    route_api(page, responses["verified"])
    res = call_ask(page, SHOTS_Q)
    assert res["ok"] and res["data"] == responses["verified"]


@pytest.mark.parametrize("text,fragment", [("", "type a question"), ("   \n\t ", "type a question"),
                                           ("x" * 1001, "at most 1000")], ids=["empty", "whitespace", "too-long"])
def test_invalid_questions_are_rejected_before_sending(page, text, fragment):
    log = []
    route_api(page, {}, log=log)
    res = call_ask(page, text)
    assert not res["ok"] and res["kind"] == "validation" and fragment in res["message"] and log == []


def test_backend_validation_error_is_handled(page, responses):
    route_api(page, responses["422"], status=422)
    res = call_ask(page, SHOTS_Q)
    assert res["kind"] == "http" and res["status"] == 422 and "not accepted" in res["message"]
    assert res["requestId"] == responses["422"]["request_id"]


def test_server_error_is_handled_without_leaking(page, responses):
    route_api(page, responses["500"], status=500)
    res = call_ask(page, SHOTS_Q)
    assert res["kind"] == "http" and res["status"] == 500 and responses["500"]["request_id"] in res["message"]
    assert "gsk_secret" not in json.dumps(res)


def test_network_failure_is_handled(page):
    page.route(f"{FAKE}/api/ask", lambda r: r.abort("connectionrefused"))
    res = call_ask(page, SHOTS_Q)
    assert res["kind"] == "network" and FAKE in res["message"] and "CORS_ORIGINS" in res["message"]


@pytest.mark.parametrize("body,kind", [
    ("<html>oops</html>", "parse"),
    ('{"outcome": "verified"', "parse"),
    ("[]", "contract"),
])
def test_unreadable_response_is_handled(page, body, kind):
    route_api(page, body)
    assert call_ask(page, SHOTS_Q)["kind"] == kind


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(outcome="excellent"),
    lambda d: d.update(answer="   "),
    lambda d: d.pop("grounding"),
    lambda d: d.update(verified=True, outcome="fallback"),
    lambda d: d.update(grounding={**d["grounding"], "passed": False}),       # verified but failed check
], ids=["unknown-outcome", "blank-answer", "no-grounding", "verified-flag-mismatch", "passed-mismatch"])
def test_inconsistent_response_is_refused(page, responses, mutate):
    body = copy.deepcopy(responses["verified"])
    mutate(body)
    route_api(page, body)
    res = call_ask(page, SHOTS_Q)
    assert not res["ok"] and res["kind"] == "contract" and res["requestId"] == body["request_id"]


def test_timeout_and_duplicate_submission(page, responses):
    held = []
    hold_api(page, held)
    res = call_ask(page, SHOTS_Q, timeoutMs=300)
    assert res["kind"] == "timeout"
    page.evaluate("() => { window.__first = AgentApi.ask('q one'); window.__first.catch(() => {}); }")
    page.wait_for_function("AgentApi.isBusy()")
    second = call_ask(page, "q two")
    assert second["kind"] == "busy"
    page.wait_for_timeout(100)
    held[-1].fulfill(status=200, content_type="application/json", body=json.dumps(responses["verified"]))
    assert page.evaluate("window.__first.then(d => d.outcome)") == "verified"
    assert len(held) == 2                                             # timed-out one + the first; never the duplicate
    release(held[:1])


def test_logs_do_not_contain_question_or_body(page, responses):
    messages = []
    page.on("console", lambda m: messages.append(m.text))
    route_api(page, responses["500"], status=500)
    call_ask(page, "purple-unicorn-marker question")
    page.wait_for_timeout(50)
    text = " ".join(messages)
    assert "[agent] request failed" in text and "purple-unicorn-marker" not in text


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def test_verified_answer_with_evidence_and_warnings(page, responses):
    d = responses["covid"]
    route_api(page, d)
    ui_ask(page, COVID_Q)
    card = page.locator("#agentResult .agent-result")
    assert card.get_attribute("data-outcome") == "verified"
    assert page.inner_text(".agent-badge") == "✓ Verified"
    assert page.inner_text(".agent-answer") == d["answer"]
    warnings = page.locator(".agent-warnings li")
    assert warnings.count() == len(d["warnings"]) > 0 and page.locator(".agent-warnings").is_visible()
    assert "2018-19 is incomplete" in page.inner_text(".agent-warnings")
    assert page.locator(".agent-details").get_attribute("open") is None          # diagnostics collapsed
    assert d["request_id"] in page.text_content(".agent-details")                # present, just collapsed
    assert "Answered in" in page.inner_text("#agentProgress") and page.errors == []


def test_key_stats_and_chart_suggestions_are_shown_not_applied(page, responses):
    d = responses["retry"]
    route_api(page, d)
    ui_ask(page, SHOTS_Q)
    assert page.inner_text(".agent-stat-value") == "0.597"
    link = page.locator(".agent-result a")
    assert link.inner_text() == "Attack (Q2) · x = AvgShots · all leagues"
    assert link.get_attribute("href") == "q2-offensive-vs-success.html"
    assert page.url.endswith("/frontend/home.html")                              # nothing navigated
    page.click(".agent-details summary")
    assert re.search(r"Retries\s+1\b", page.inner_text(".agent-details"))


def test_no_evidence_is_honest(page, responses):
    d = responses["no_evidence"]
    route_api(page, d)
    ui_ask(page, demo.SCENARIOS["no_evidence"][0])
    assert page.inner_text(".agent-badge") == "No evidence"
    assert "could not answer" in page.inner_text(".agent-outcome-note")
    assert page.inner_text(".agent-answer") == d["answer"] and page.locator(".agent-stat").count() == 0
    assert "Unknown team 'Arsenl'" in page.inner_text(".agent-result")


def test_fallback_is_labelled_and_never_verified(page, responses):
    d = responses["fallback"]
    route_api(page, d)
    ui_ask(page, SHOTS_Q)
    assert page.inner_text(".agent-badge") == "Fallback"
    assert "Verified" not in page.inner_text(".agent-outcome")
    assert demo.SHOTS_BAD["answer"] not in page.inner_text("#agentPanel")


def test_unverified_is_not_presented_as_confirmed(page, responses):
    d = responses["unverified"]
    route_api(page, d)
    ui_ask(page, SHOTS_Q)
    assert page.locator(".agent-result--unverified").count() == 1
    assert page.inner_text(".agent-badge") == "⚠ Unverified" and "Verified" not in page.inner_text(".agent-badge").replace("Unverified", "")
    page.click(".agent-details summary")
    assert "unsupported_number: 0.9" in page.inner_text(".agent-details")
    assert "could not be fully verified" in page.inner_text(".agent-explanation")


def test_missing_optional_fields_still_render(page):
    minimal = {"request_id": "r1", "question": SHOTS_Q, "outcome": "verified", "verified": True,
               "answer": "Shots and points are associated.", "grounding": {"passed": True}}
    route_api(page, minimal)
    ui_ask(page, SHOTS_Q)
    assert page.inner_text(".agent-answer") == minimal["answer"]
    assert page.locator(".agent-warnings, .agent-stat").count() == 0 and page.errors == []


def test_inconsistent_response_shows_safe_error_not_answer(page, responses):
    body = {**responses["verified"], "grounding": {**responses["verified"]["grounding"], "passed": False}}
    route_api(page, body)
    ui_ask(page, SHOTS_Q)
    assert page.locator(".agent-result").count() == 0
    assert page.locator(".agent-error").get_attribute("data-kind") == "contract"
    assert body["request_id"] in page.inner_text(".agent-error") and body["answer"] not in page.inner_text("#agentPanel")


def test_backend_text_is_never_executed(page, responses):
    d = copy.deepcopy(responses["retry"])
    d.update(answer=f"Answer {XSS}", evidence=[XSS], warnings=[XSS], explanation=XSS,
             key_stats=[{"label": XSS, "value": XSS, "source": "offensive-1"}],
             errors=[{"node": XSS, "kind": "x", "message": XSS}])
    d["tool_results"][0]["args"] = {"x_metric": XSS}
    d["chart_actions"] = [{"page": "q2", "x_metric": XSS}]
    route_api(page, d)
    ui_ask(page, SHOTS_Q)
    page.click(".agent-details summary")
    page.wait_for_timeout(100)
    assert page.evaluate("window.__pwned") is None
    assert page.locator("#agentPanel img, #agentPanel script").count() == 0
    assert page.inner_text(".agent-answer") == f"Answer {XSS}"


def test_ui_ignores_double_submit_and_supports_cancel(page, responses):
    held = []
    hold_api(page, held)
    page.click("#agentLauncher")
    page.fill("#agentQuestion", SHOTS_Q)
    page.click("#agentSubmit")
    page.wait_for_function("AgentPanel.isBusy()")
    assert page.is_disabled("#agentSubmit") and page.locator(".agent-chip").first.is_disabled()
    page.evaluate("AgentPanel.submit('another question')")             # programmatic second submit
    page.press("#agentQuestion", "Control+Enter")                       # keyboard second submit
    page.wait_for_timeout(200)
    assert len(held) == 1
    page.click("#agentCancel")
    page.wait_for_selector(".agent-error[data-kind='cancelled']")
    assert not page.is_disabled("#agentSubmit")
    release(held)


def test_example_chip_sends_a_real_request_and_validation_message_shows(page, responses):
    log = []
    route_api(page, responses["covid"], log=log)
    page.click("#agentLauncher")
    page.fill("#agentQuestion", "   ")
    page.click("#agentSubmit")
    assert "type a question" in page.inner_text("#agentFormError") and log == []
    page.click(f".agent-chip:has-text('{COVID_Q}')")
    page.wait_for_selector(".agent-result")
    assert len(log) == 1 and log[0].post_data_json == {"question": COVID_Q}
    assert page.input_value("#agentQuestion") == COVID_Q


def test_service_status_is_checked_once_and_throttled(page):
    log = []
    route_api(page, {"status": "ready", "checks": {"graph": "ok", "dataset": "ok (57787 matches)",
                                                    "llm": "offline-mock (not called by this check)"}},
              path="/health/ready", log=log)
    page.click("#agentLauncher")
    page.wait_for_function("document.querySelector('#agentStatus').dataset.state === 'ready'")
    assert page.inner_text("#agentStatus") == "Online · offline-mock"
    page.click(".agent-close")
    page.click("#agentLauncher")
    page.wait_for_timeout(200)
    assert len(log) == 1


def test_offline_backend_is_reported_with_help(page):
    page.route(f"{FAKE}/**", lambda r: r.abort("connectionrefused"))
    ui_ask(page, SHOTS_Q)
    assert page.locator(".agent-error").get_attribute("data-kind") == "network"
    assert "uvicorn app.main:create_app" in page.inner_text(".agent-error")
    page.wait_for_function("document.querySelector('#agentStatus').dataset.state === 'offline'")
    assert page.locator(".agent-error button:has-text('Check service')").count() == 1


# ---------------------------------------------------------------------------
# Integration with a real backend (offline-mock LLM) and the real dashboard
# ---------------------------------------------------------------------------
def test_end_to_end_with_real_backend(browser, site):
    with backend(cors_origins=site) as api:
        p = _new_page(browser, site, api_base=api)
        try:
            p.click("#agentLauncher")
            p.wait_for_function("document.querySelector('#agentStatus').dataset.state === 'ready'")
            p.click(f".agent-chip:has-text('{SHOTS_Q}')")
            p.wait_for_selector(".agent-result", timeout=30000)
            assert p.inner_text(".agent-badge") == "✓ Verified" and "r = 0.597" in p.inner_text(".agent-answer")
            ui_ask(p, COVID_Q)                                           # a second, different question
            p.wait_for_function("document.querySelector('.agent-answer').textContent.includes('45.4%')")
            assert "2018-19 is incomplete" in p.inner_text(".agent-warnings")
            assert p.errors == []
        finally:
            p.context.close()


def test_origin_not_in_cors_list_is_blocked_and_explained(browser, site):
    with backend(cors_origins="http://localhost:5500") as api:          # not this page's origin
        p = _new_page(browser, site, api_base=api)
        try:
            ui_ask(p, SHOTS_Q)
            assert p.locator(".agent-error").get_attribute("data-kind") == "network"
            assert "CORS_ORIGINS" in p.inner_text(".agent-error")
        finally:
            p.context.close()


def test_assistant_does_not_interfere_with_d3_dashboard(browser, site, responses):
    p = _new_page(browser, site, path="frontend/q2-offensive-vs-success.html")
    try:
        route_api(p, responses["verified"])
        p.wait_for_selector("#q2Chart circle.scatter-dot", timeout=60000)
        before = p.locator("#q2Chart circle.scatter-dot").count()
        ui_ask(p, SHOTS_Q)
        assert p.inner_text(".agent-badge") == "✓ Verified"
        assert p.locator("#q2Chart circle.scatter-dot").count() == before > 0
        p.select_option("#q2LeagueFilter", "Serie A")                    # the chart still reacts to its controls
        p.wait_for_function(f"document.querySelectorAll('#q2Chart circle.scatter-dot').length < {before}")
        assert p.errors == []
    finally:
        p.context.close()


def test_all_pages_load_the_panel_without_errors(browser, site):
    for path in ("frontend/home.html", "frontend/index.html", "frontend/q2-offensive-vs-success.html",
                 "frontend/q3-disciplinary.html"):
        p = _new_page(browser, site, path=path)
        try:
            assert p.locator("#agentLauncher").count() == 1 and p.locator("#agentPanel").is_hidden()
            assert p.evaluate("typeof AgentApi.ask") == "function" and p.errors == []
        finally:
            p.context.close()
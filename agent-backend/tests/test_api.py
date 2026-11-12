"""
Stage 8: the FastAPI service. In-process TestClient only: no server, no network, no API key.
The default app runs the real graph with the offline mock LLM; other tests inject a graph
(a real one with scripted MockLLMs, or a stub returning a hand-made final state).
"""
from __future__ import annotations

import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import tools.common
import tools.registry as registry
from agents import demo
from agents import graph as agent_graph
from agents import guard as G
from agents.analysts import ToolRequest, check_request
from agents.router import AGENT_TOOLS
from agents.state import MAX_QUESTION_CHARS
from app import schemas
from app.config import Settings
from app.main import API_VERSION, MAX_BODY_BYTES, create_app
from app.offline import OfflineLLM, choose_tool, offline_llm
from llm.mock import MockLLM

SHOTS_Q = "Do teams that take more shots get more points?"
LOCAL = "http://localhost:63342"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, preload_data=False, **kw)


def client_for(graph=None, **kw) -> TestClient:
    return TestClient(create_app(settings(**kw), graph=graph))


class StubGraph:
    """Stands in for the compiled graph: returns a prepared final state (or raises)."""
    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error, self.calls = result, error, 0

    def invoke(self, state, config=None):
        self.calls += 1
        if self.error:
            raise self.error
        out = copy.deepcopy(self.result)
        out.setdefault("question", state["question"])
        return out


@pytest.fixture(scope="module")
def client() -> TestClient:
    return client_for()


@pytest.fixture(scope="module")
def verified_state() -> dict:
    """A real final state from the real graph (offline LLM, real tools)."""
    return agent_graph.ask(agent_graph.build_graph(offline_llm()), SHOTS_Q)


def scenario_client(name: str) -> TestClient:
    return client_for(agent_graph.build_graph(MockLLM(demo.SCENARIOS[name][1])))


def is_problem(r, status: int) -> dict:
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert body["status"] == status and body["title"] and body["detail"] and body["request_id"]
    return body


# ---------------------------------------------------------------------------
# A  application and health
# ---------------------------------------------------------------------------
def test_app_metadata_routes_and_docs(client):
    app = client.app
    assert app.title == "Football Analytics Agent API" and app.version == API_VERSION
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/health/ready", "/api/ask"} <= set(paths)
    assert "post" in paths["/api/ask"] and set(paths["/api/ask"]["post"]["responses"]) >= {"200", "422", "500"}
    assert client.get("/docs").status_code == 200


def test_health_returns_documented_body_and_touches_nothing(monkeypatch):
    monkeypatch.setattr(registry, "call_tool", lambda *a, **k: pytest.fail("health ran a tool"))
    stub = StubGraph(error=AssertionError("health invoked the graph"))
    r = client_for(stub).get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "service": "football-analytics-api",
                                                 "version": API_VERSION}
    assert stub.calls == 0 and r.headers["x-request-id"]


def test_ready_checks_graph_and_data_without_llm_or_tools(monkeypatch):
    monkeypatch.setattr(registry, "call_tool", lambda *a, **k: pytest.fail("readiness ran a tool"))
    stub = StubGraph(error=AssertionError("readiness invoked the graph"))
    r = client_for(stub).get("/health/ready")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ready" and stub.calls == 0
    assert body["checks"]["graph"] == "ok" and body["checks"]["dataset"].startswith("ok (")
    assert "not called" in body["checks"]["llm"]


def test_ready_is_503_when_data_is_unavailable(monkeypatch):
    def broken():
        raise FileNotFoundError("/secret/path/datasets missing")
    monkeypatch.setattr(tools.common, "get_dataset", broken)
    r = client_for(StubGraph()).get("/health/ready")
    assert r.status_code == 503 and r.json()["checks"]["dataset"] == "unavailable"
    assert "/secret/path" not in r.text


def test_real_llm_mode_is_refused_until_it_exists():
    with pytest.raises(RuntimeError, match="USE_MOCK_LLM"):
        create_app(settings(use_mock_llm=False))


@pytest.mark.parametrize("origins", ["*", "http://localhost:63342/app", "localhost:63342", "http://a.com,*"])
def test_settings_reject_wildcard_or_malformed_origins(origins):
    with pytest.raises(ValueError):
        settings(cors_origins=origins)


def test_graph_is_built_once_and_reused(monkeypatch):
    built = []
    real = agent_graph.build_graph
    monkeypatch.setattr(agent_graph, "build_graph", lambda *a, **k: built.append(1) or real(*a, **k))
    c = client_for()
    for _ in range(2):
        assert c.post("/api/ask", json={"question": SHOTS_Q}).status_code == 200
    assert len(built) == 1


# ---------------------------------------------------------------------------
# B  request validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("body", [
    {}, {"question": ""}, {"question": "   \n\t "}, {"question": 42}, {"question": None},
    {"question": ["shots?"]}, {"question": True}, {"question": "x" * (MAX_QUESTION_CHARS + 1)},
    {"question": SHOTS_Q, "thread_id": "abc"},                     # unsupported fields are rejected
])
def test_invalid_bodies_are_422_problems(body):
    stub = StubGraph(error=AssertionError("graph must not run for an invalid request"))
    r = client_for(stub).post("/api/ask", json=body)
    problem = is_problem(r, 422)
    assert problem["errors"] and stub.calls == 0
    assert "x" * 100 not in r.text                                 # the input is not echoed back


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"just a string"'])
def test_malformed_json_is_422(raw):
    r = client_for(StubGraph()).post("/api/ask", content=raw, headers={"Content-Type": "application/json"})
    is_problem(r, 422)


def test_oversized_body_is_413():
    r = client_for(StubGraph()).post("/api/ask", content=b'{"question": "' + b"a" * MAX_BODY_BYTES + b'"}',
                                     headers={"Content-Type": "application/json"})
    is_problem(r, 413)


def test_question_is_normalised_and_limit_is_inclusive(verified_state):
    stub = StubGraph({**verified_state, "question": None})
    stub.result.pop("question")
    c = client_for(stub)
    r = c.post("/api/ask", json={"question": "  Do teams   take more shots?\n"})
    assert r.status_code == 200 and r.json()["question"] == "Do teams take more shots?"
    assert c.post("/api/ask", json={"question": "q" * MAX_QUESTION_CHARS}).status_code == 200


# ---------------------------------------------------------------------------
# C  successful requests
# ---------------------------------------------------------------------------
def test_verified_answer_is_serialised_faithfully(client, verified_state):
    r = client.post("/api/ask", json={"question": SHOTS_Q}, headers={"X-Request-ID": "test-123"})
    assert r.status_code == 200
    j = r.json()
    assert j["request_id"] == "test-123" == r.headers["x-request-id"]
    assert j["outcome"] == "verified" and j["verified"] is True and j["grounding"]["passed"] is True
    assert j["answer"] == verified_state["draft_answer"]["answer"] and "r = 0.597" in j["answer"]
    assert j["retries"] == 0 and j["intent"] == "offensive" and j["analysts"] == ["offensive"]
    [t] = j["tool_results"]
    assert (t["call_id"], t["analyst"], t["tool"], t["ok"]) == ("offensive-1", "offensive", "correlation_analysis", True)
    assert t["result"]["r"] == registry.call_tool("correlation_analysis", {"x_metric": "AvgShots"})["result"]["r"]
    assert j["warnings"] and len(j["warnings"]) == len(set(j["warnings"]))
    assert any("2018-19 is incomplete" in w for w in j["warnings"])
    assert j["usage"]["models"] == ["offline-mock"] and j["usage"]["llm_calls"] == 3
    assert [n["node"] for n in j["trace"]] == ["router", "analyst_offensive", "synthesiser", "guard", "finalise"]
    assert r.headers["server-timing"].endswith(f"total;dur={j['duration_ms']:.1f}")
    assert "analyst_offensive;dur=" in r.headers["server-timing"]


def test_api_returns_exactly_what_the_existing_graph_produces(client, verified_state, monkeypatch):
    """The endpoint calls agents.graph.ask on the shared graph; it does not run its own workflow."""
    seen = []
    real_ask = agent_graph.ask
    monkeypatch.setattr(agent_graph, "ask", lambda app, q, **k: seen.append((app, q)) or real_ask(app, q, **k))
    j = client.post("/api/ask", json={"question": SHOTS_Q}).json()
    assert seen == [(client.app.state.graph, SHOTS_Q)]
    assert j["answer"] == verified_state["draft_answer"]["answer"]
    assert [t["result"] for t in j["tool_results"]] == [r["result"] for r in verified_state["tool_results"]]
    assert j["grounding"]["checked"] == verified_state["grounding_result"]["checked"]


def test_retry_key_stats_and_chart_actions_are_preserved():
    j = scenario_client("retry").post("/api/ask", json={"question": SHOTS_Q}).json()
    assert j["outcome"] == "verified" and j["retries"] == 1
    assert j["key_stats"] == [{"label": "Pearson r, shots vs points", "value": "0.597", "source": "offensive-1"}]
    assert j["chart_actions"] == [{"page": "q2", "league": "all", "x_metric": "AvgShots"}]
    assert [e["kind"] for e in j["errors"]] == ["grounding_rejected"]
    assert [n["node"] for n in j["trace"]].count("synthesiser") == 2


def test_two_analysts_both_reported():
    q = demo.SCENARIOS["two_analysts"][0]
    j = scenario_client("two_analysts").post("/api/ask", json={"question": q}).json()
    assert sorted(t["call_id"] for t in j["tool_results"]) == ["discipline-1", "offensive-1"]
    assert j["analysts"] == ["offensive", "discipline"]


def test_missing_optional_telemetry_does_not_break_the_response(verified_state):
    minimal = {k: verified_state[k] for k in ("question", "draft_answer", "grounding_result", "outcome")}
    r = client_for(StubGraph(minimal)).post("/api/ask", json={"question": SHOTS_Q})
    j = r.json()
    assert r.status_code == 200 and j["outcome"] == "verified"
    assert j["tool_results"] == j["errors"] == j["trace"] == j["warnings"] == j["chart_actions"] == []
    assert j["usage"]["llm_calls"] == 0 and j["retries"] == 0 and j["intent"] is None
    assert r.headers["server-timing"].startswith("total;dur=")


# ---------------------------------------------------------------------------
# D  non-verified outcomes
# ---------------------------------------------------------------------------
def test_no_evidence_is_a_safe_200_without_statistics():
    q = demo.SCENARIOS["no_evidence"][0]
    j = scenario_client("no_evidence").post("/api/ask", json={"question": q}).json()
    assert j["outcome"] == "no_evidence" and j["verified"] is False
    assert "could not establish" in j["answer"] and j["key_stats"] == [] and j["usage"]["llm_calls"] == 2
    assert j["tool_results"][0]["ok"] is False and j["tool_results"][0]["result"] is None
    assert j["tool_results"][0]["error"]["code"] == "unknown_team" and "Arsenal" in j["tool_results"][0]["error"]["suggestions"]
    assert ("synthesiser", "no_evidence") in {(e["node"], e["kind"]) for e in j["errors"]}


def test_fallback_is_identified_and_rejected_draft_is_not_returned():
    j = scenario_client("fallback").post("/api/ask", json={"question": SHOTS_Q}).json()
    assert j["outcome"] == "fallback" and j["verified"] is False and j["grounding"]["passed"] is True
    assert demo.SHOTS_BAD["answer"] not in json.dumps(j) and "0.9)" not in j["answer"]
    assert j["chart_actions"] == [] and [e["kind"] for e in j["errors"]] == ["grounding_rejected", "grounding_failed"]


def test_unverified_stays_unverified(monkeypatch):
    real = G.check
    monkeypatch.setattr(G, "check", lambda s: {**real(s), "passed": False,
                                               "issues": [{"kind": "x", "field": "answer", "claim": "c", "detail": "d"}]})
    j = client_for().post("/api/ask", json={"question": SHOTS_Q}).json()
    assert j["outcome"] == "unverified" and j["verified"] is False and j["grounding"]["passed"] is False
    assert j["grounding"]["issues"] and "could not be fully verified" in j["explanation"]


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(outcome="verified", grounding_result={**s["grounding_result"], "passed": False}),
    lambda s: s.update(outcome="unverified"),                          # but grounding passed=True
    lambda s: s.pop("outcome"),
    lambda s: s.update(outcome="excellent"),
    lambda s: s.update(draft_answer=None),
    lambda s: s.update(draft_answer={**s["draft_answer"], "answer": "  "}),
    lambda s: s.update(grounding_result=None),
    lambda s: s.update(grounding_result={**s["grounding_result"], "passed": "yes"}),
    lambda s: s.update(tool_results=[{"call_id": "x"}]),               # malformed telemetry
], ids=["verified-but-failed", "unverified-but-passed", "no-outcome", "bad-outcome", "no-draft", "blank-answer",
        "no-grounding", "non-bool-passed", "bad-tool-result"])
def test_inconsistent_or_incomplete_graph_output_is_500_not_an_answer(verified_state, mutate):
    s = copy.deepcopy(verified_state)
    mutate(s)
    r = client_for(StubGraph(s)).post("/api/ask", json={"question": SHOTS_Q})
    body = is_problem(r, 500)
    assert body["title"] == "Incomplete analysis result"
    assert "answer" not in body and "verified" not in body and "0.597" not in r.text


# ---------------------------------------------------------------------------
# E  failures and leakage
# ---------------------------------------------------------------------------
SECRET = "GROQ_API_KEY=gsk_live_ABC123 in C:\\Users\\Admin\\secret.py and /home/admin/.env"


def test_graph_exception_is_a_safe_500(caplog):
    with caplog.at_level(logging.ERROR, logger="football_api"):
        r = client_for(StubGraph(error=RuntimeError(SECRET))).post("/api/ask", json={"question": SHOTS_Q})
    body = is_problem(r, 500)
    assert body["title"] == "Analysis failed" and "verified" not in body
    for leak in ("gsk_live", "Traceback", "C:\\", "/home/admin", "RuntimeError", "secret.py"):
        assert leak not in r.text
    assert any("RuntimeError" in rec.getMessage() for rec in caplog.records)   # still diagnosable server-side
    assert not any("gsk_live" in rec.getMessage() for rec in caplog.records)


def test_node_exception_text_is_not_exposed():
    script = {**demo.SCENARIOS["verified"][1], "synthesiser": lambda req: (_ for _ in ()).throw(RuntimeError(SECRET))}
    r = client_for(agent_graph.build_graph(MockLLM(script))).post("/api/ask", json={"question": SHOTS_Q})
    j = r.json()
    assert r.status_code == 200 and {"node": "synthesiser", "kind": "node_exception", "message": "RuntimeError"} in j["errors"]
    assert "gsk_live" not in r.text and "secret.py" not in r.text


@pytest.mark.parametrize("kind,message,expected", [
    ("llm_error", "LLMRateLimitError: 429 key gsk_abc", "LLMRateLimitError"),
    ("guard_error", "ZeroDivisionError: division by zero", "ZeroDivisionError"),
    ("tool_error", "team_season_stats: unknown_team: Unknown team 'Arsenl'.", "team_season_stats: unknown_team: Unknown team 'Arsenl'."),
    ("bad_output", "ValueError: api_key=abc123 at /home/u/x.py", "ValueError: [redacted] at [redacted]"),
])
def test_public_error_messages(kind, message, expected):
    assert schemas._public_message(kind, message) == expected


def test_unknown_route_and_wrong_method_are_problems(client):
    is_problem(client.get("/api/nope"), 404)
    is_problem(client.get("/api/ask"), 405)


def test_logs_do_not_contain_the_question(caplog):
    q = "Do teams with purple-unicorn-marker shots get more points?"
    with caplog.at_level(logging.INFO, logger="football_api"):
        r = client_for().post("/api/ask", json={"question": q})
    assert r.status_code == 200
    text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "outcome=verified" in text and "purple-unicorn-marker" not in text


# ---------------------------------------------------------------------------
# F  configuration, CORS, concurrency, offline LLM
# ---------------------------------------------------------------------------
def test_no_api_key_or_real_llm_needed(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    c = client_for()
    assert c.app.state.llm_mode == "offline-mock"
    assert c.post("/api/ask", json={"question": SHOTS_Q}).json()["usage"]["models"] == ["offline-mock"]


@pytest.mark.parametrize("origin", [LOCAL, "http://127.0.0.1:63342"])
def test_cors_preflight_allows_local_dashboard(client, origin):
    r = client.options("/api/ask", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin
    assert "POST" in r.headers["access-control-allow-methods"]
    assert "access-control-allow-credentials" not in r.headers


def test_cors_rejects_unconfigured_origin(client):
    r = client.options("/api/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers and "timing-allow-origin" not in r.headers


def test_cors_origins_are_configurable():
    c = client_for(StubGraph(), cors_origins="http://localhost:5500")
    ok = c.get("/health", headers={"Origin": "http://localhost:5500"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5500"
    assert ok.headers["timing-allow-origin"] == "http://localhost:5500"
    assert "server-timing" in ok.headers["access-control-expose-headers"].lower()
    assert "access-control-allow-origin" not in c.get("/health", headers={"Origin": LOCAL}).headers


def test_concurrent_requests_do_not_mix_results(client):
    questions = {"Do teams that take more shots get more points?": "AvgShots",
                 "Do yellow cards cost teams points?": "AvgYellowCards",
                 "Do fouls cost teams points in Serie A?": "AvgFouls",
                 "Do goals scored relate to points in La Liga?": "GoalsFor"}
    jobs = list(questions) * 4

    def post(q):
        return q, client.post("/api/ask", json={"question": q}).json()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(post, jobs))
    assert len({j["request_id"] for _, j in results}) == len(jobs)
    for q, j in results:
        assert j["question"] == q and j["outcome"] == "verified"
        assert [t["args"]["x_metric"] for t in j["tool_results"]] == [questions[q]]
        assert len(j["tool_results"]) == 1 and [e for e in j["errors"]] == []


def test_offline_llm_keeps_no_call_log():
    llm = offline_llm()
    assert isinstance(llm, OfflineLLM)
    agent_graph.ask(agent_graph.build_graph(llm), SHOTS_Q)
    assert llm.calls == []


@pytest.mark.parametrize("question", [SHOTS_Q, "Was home advantage lower during COVID in Serie A?",
                                      "Who won the Premier League in 2004-05?", "Do red cards matter in La Liga?",
                                      "Compare Arsenal and Chelsea on yellow cards and shots", "hello"])
def test_offline_tool_choices_always_pass_analyst_validation(question):
    for agent in AGENT_TOOLS:
        call = choose_tool(agent, question)
        args, problem = check_request(agent, ToolRequest(**call))
        assert problem is None, (agent, question, problem)
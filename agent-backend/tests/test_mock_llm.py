"""Stage 1: the mock LLM. No API key, no network, fully deterministic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from llm.base import (LLMClient, LLMOutputError, LLMRateLimitError, LLMTimeoutError, Message,
                      Usage)
from llm.mock import (MockLLM, MockScriptError, count_tokens, keyword_responder,
                      when_prompt_contains)


def ask(llm, text, purpose, json_mode=True, system="You are a test."):
    return llm.complete([Message("system", system), Message("user", text)], purpose=purpose, json_mode=json_mode)


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
def test_implements_llm_client_protocol():
    assert isinstance(MockLLM(), LLMClient)


def test_dict_reply_is_json_and_parses():
    llm = MockLLM({"router": {"intent": "trend", "agents": ["trend"]}})
    r = ask(llm, "How did Arsenal do?", "router")
    assert r.json() == {"intent": "trend", "agents": ["trend"]}
    assert r.content == '{"agents": ["trend"], "intent": "trend"}'   # sorted keys -> stable text
    assert r.model == "mock-llm" and r.purpose == "router"


def test_single_response_repeats_forever():
    llm = MockLLM({"router": {"intent": "trend"}})
    assert [ask(llm, "q", "router").json() for _ in range(3)] == [{"intent": "trend"}] * 3


def test_queue_is_consumed_in_order_then_errors():
    llm = MockLLM({"synthesiser": ["first", "second"]})
    assert ask(llm, "q", "synthesiser", json_mode=False).content == "first"
    assert ask(llm, "q", "synthesiser", json_mode=False).content == "second"
    with pytest.raises(MockScriptError, match="exhausted"):
        ask(llm, "q", "synthesiser")


def test_unscripted_purpose_fails_loudly():
    llm = MockLLM({"router": "x"})
    with pytest.raises(MockScriptError, match="no response scripted for purpose 'guard'"):
        ask(llm, "q", "guard")
    assert llm.calls[-1].error is not None


def test_default_response():
    llm = MockLLM(default={"ok": True})
    assert ask(llm, "q", "anything").json() == {"ok": True}


# ---------------------------------------------------------------------------
# Purpose matching
# ---------------------------------------------------------------------------
def test_exact_purpose_beats_prefix_beats_wildcard():
    llm = MockLLM({"analyst:trend": "exact", "analyst": "prefix", "*": "wildcard"})
    assert ask(llm, "q", "analyst:trend", json_mode=False).content == "exact"
    assert ask(llm, "q", "analyst:offensive", json_mode=False).content == "prefix"
    assert ask(llm, "q", "synthesiser", json_mode=False).content == "wildcard"


def test_per_purpose_queues_are_independent():
    llm = MockLLM({"analyst:trend": ["t1", "t2"], "analyst:offensive": ["o1", "o2"]})
    assert ask(llm, "q", "analyst:offensive", False).content == "o1"
    assert ask(llm, "q", "analyst:trend", False).content == "t1"
    assert ask(llm, "q", "analyst:offensive", False).content == "o2"


# ---------------------------------------------------------------------------
# Simulating the agent roles
# ---------------------------------------------------------------------------
ROUTER = keyword_responder(
    [(("shot", "shoot", "goal", "attack"), {"intent": "offensive", "agents": ["offensive"]}),
     (("card", "foul", "red", "yellow"), {"intent": "discipline", "agents": ["discipline"]}),
     (("significant", "home advantage", "covid"), {"intent": "hypothesis", "agents": ["trend"]})],
    default={"intent": "general_statistics", "agents": ["trend"]},
)


@pytest.mark.parametrize("question,intent", [
    ("Do teams that shoot more get more points?", "offensive"),
    ("Do yellow cards hurt league position?", "discipline"),
    ("Was home advantage significantly lower during COVID?", "hypothesis"),
    ("Who won the league in 2004-05?", "general_statistics"),
    ("How many goals has Arsenal scored?", "offensive"),     # 'red' must not match 'scored'
])
def test_router_decisions(question, intent):
    llm = MockLLM({"router": ROUTER})
    assert ask(llm, question, "router").json()["intent"] == intent


def test_analyst_response_can_depend_on_request():
    def analyst(req):
        team = "Arsenal" if "Arsenal" in req.last_user else "Chelsea"
        return {"tool_calls": [{"tool": "season_trend", "args": {"team": team, "metric": "Points"}}]}
    llm = MockLLM({"analyst": analyst})
    assert ask(llm, "Arsenal points trend", "analyst:trend").json()["tool_calls"][0]["args"]["team"] == "Arsenal"


def test_grounding_failure_then_success_via_queue():
    """Draft 1 invents a number; draft 2 is grounded. The graph's retry loop will consume both."""
    bad = {"answer": "Arsenal scored 999 goals."}
    good = {"answer": "Arsenal scored 73 goals."}
    llm = MockLLM({"synthesiser": [bad, good]})
    assert ask(llm, "draft", "synthesiser").json() == bad
    assert ask(llm, "draft + feedback", "synthesiser").json() == good
    assert len(llm.calls_for("synthesiser")) == 2


def test_model_corrects_itself_when_feedback_is_in_prompt():
    llm = MockLLM({"synthesiser": when_prompt_contains("GROUNDING FEEDBACK",
                                                       then={"answer": "73 goals"},
                                                       otherwise={"answer": "999 goals"})})
    assert ask(llm, "write the answer", "synthesiser").json()["answer"] == "999 goals"
    assert ask(llm, "write the answer\nGROUNDING FEEDBACK: 999 not in tool results",
               "synthesiser").json()["answer"] == "73 goals"


def test_model_that_never_grounds_lets_us_test_retry_limit():
    llm = MockLLM({"synthesiser": {"answer": "999 goals"}})   # always wrong
    answers = {ask(llm, "draft", "synthesiser").json()["answer"] for _ in range(5)}
    assert answers == {"999 goals"} and len(llm.calls) == 5


# ---------------------------------------------------------------------------
# Failure injection
# ---------------------------------------------------------------------------
def test_rate_limit_then_success():
    llm = MockLLM({"router": [LLMRateLimitError("429"), {"intent": "trend"}]})
    with pytest.raises(LLMRateLimitError):
        ask(llm, "q", "router")
    assert ask(llm, "q", "router").json() == {"intent": "trend"}
    assert llm.calls[0].error is not None and llm.calls[1].response is not None


def test_timeout_from_callable():
    llm = MockLLM({"router": lambda req: LLMTimeoutError("slow")})
    with pytest.raises(LLMTimeoutError):
        ask(llm, "q", "router")


def test_malformed_json_is_detectable():
    llm = MockLLM({"router": "I think it's about shots"})       # model ignored JSON mode
    r = ask(llm, "q", "router")
    with pytest.raises(LLMOutputError) as e:
        r.json()
    assert e.value.content == "I think it's about shots"


# ---------------------------------------------------------------------------
# Usage, latency, determinism
# ---------------------------------------------------------------------------
def test_token_counting_is_deterministic():
    assert count_tokens("") == 0 and count_tokens("abcd") == 1 and count_tokens("abcde") == 2
    llm = MockLLM({"router": {"intent": "trend"}})
    r = ask(llm, "How did Arsenal do?", "router", system="sys")
    expected_prompt = (count_tokens("sys") + 4) + (count_tokens("How did Arsenal do?") + 4)
    assert r.usage == Usage(expected_prompt, count_tokens(r.content))
    assert r.usage.total_tokens == expected_prompt + count_tokens(r.content)
    assert r.latency_ms == 50.0 + 2.0 * r.usage.completion_tokens


def test_identical_runs_give_identical_results():
    def run():
        llm = MockLLM({"router": ROUTER, "synthesiser": ["a", "b"]})
        out = [ask(llm, "shots vs points", "router"), ask(llm, "x", "synthesiser", False),
               ask(llm, "x", "synthesiser", False)]
        return [(r.content, r.usage, r.latency_ms) for r in out], llm.total_usage
    assert run() == run()


def test_total_usage_sums_successful_calls_only():
    llm = MockLLM({"router": [LLMRateLimitError("429"), "ok"], "synthesiser": "done"})
    with pytest.raises(LLMRateLimitError):
        ask(llm, "q", "router", False)
    a = ask(llm, "q", "router", False)
    b = ask(llm, "q", "synthesiser", False)
    assert llm.total_usage == a.usage + b.usage


def test_reset_rewinds_queues_and_clears_log():
    llm = MockLLM({"synthesiser": ["a", "b"]})
    ask(llm, "q", "synthesiser", False)
    llm.reset()
    assert llm.calls == [] and ask(llm, "q", "synthesiser", False).content == "a"


def test_call_index_counts_per_purpose():
    llm = MockLLM({"*": lambda req: {"i": req.call_index}})
    assert [ask(llm, "q", "a").json()["i"] for _ in range(3)] == [0, 1, 2]
    assert ask(llm, "q", "b").json()["i"] == 0


def test_parallel_calls_are_safe_and_deterministic():
    """LangGraph runs analysts concurrently; separate purposes must get their own answers."""
    llm = MockLLM({f"analyst:{n}": {"agent": n} for n in ("trend", "offensive", "discipline")})
    purposes = ["analyst:trend", "analyst:offensive", "analyst:discipline"] * 20
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: (p, ask(llm, "q", p).json()["agent"]), purposes))
    assert all(p.endswith(agent) for p, agent in results)
    assert len(llm.calls) == 60


def test_no_network_needed(monkeypatch):
    """Belt and braces: block sockets entirely; the mock must still work."""
    import socket

    def boom(*a, **k):
        raise RuntimeError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", boom)
    assert ask(MockLLM(default="fine"), "q", "x", False).content == "fine"


def test_unsupported_reply_type_is_an_error():
    llm = MockLLM({"router": 42})
    with pytest.raises(TypeError):
        ask(llm, "q", "router")
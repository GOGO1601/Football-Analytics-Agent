"""
Specialist analyst nodes: trend, offensive, discipline.

An analyst never calculates football statistics. It asks the LLM WHICH registry tools to
call with WHICH arguments, checks that request, runs the approved calls through
tools.registry.call_tool, and stores the structured results in the graph state.

    make_analyst("offensive", llm) -> node(state) -> {"tool_calls", "tool_results", "token_usage", "errors"}

Safety checks on every requested call, in order:
  1. the tool must be in AGENT_TOOLS[agent] (and therefore in the registry)  -> else rejected
  2. the arguments must validate against that tool's registry schema          -> else rejected
  3. only then is the call executed, and only through registry.call_tool
The LLM never names a Python function directly and never supplies code.

If the reply is not valid JSON, or every requested call is rejected, the analyst asks the
LLM once more with the exact validation errors (a "repair" turn). Rate limits and timeouts
are recorded and the analyst returns no results; retrying those is the LLM client's job.

The three analysts share one parameterised prompt (## analyst in prompts/<version>.md)
and run independently, so LangGraph can run them in parallel: each returns only list
updates, which the state's reducers append.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import AliasChoices, BaseModel, Field, ValidationError

import tools.registry as registry
from llm.base import LLMClient, LLMError, Message
from tools.common import LEAGUES, NUMERIC_METRICS

from .prompting import render
from .router import AGENT_DESCRIPTIONS, AGENT_TOOLS
from .state import AgentError, AgentState, ToolCall, ToolResult, usage_record

MAX_TOOL_CALLS = 4      # per analyst per question
MAX_REPAIRS = 1         # extra LLM turns allowed to fix invalid JSON / rejected calls


# ---------------------------------------------------------------------------
# Tool catalogue for the prompt (generated from the registry schemas)
# ---------------------------------------------------------------------------
def _type_of(prop: dict, defs: dict) -> str:
    if "$ref" in prop:
        name = prop["$ref"].split("/")[-1]
        fields = ", ".join(defs[name]["properties"])
        return f"{{{fields}}}"
    if "anyOf" in prop:
        parts = [_type_of(p, defs) for p in prop["anyOf"] if p.get("type") != "null"]
        parts = [p for p in parts if p != "object"]  # drop the loose dict alternative if any
        return " | ".join(dict.fromkeys(parts)) or "object"
    t = prop.get("type", "any")
    if t == "array":
        return f"list[{_type_of(prop.get('items', {}), defs)}]"
    return {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}.get(t, t)


def tool_signature(name: str) -> str:
    """e.g. correlation_analysis(x_metric: str, y_metric: str = "Points", league: str = null, ...)"""
    schema = registry.TOOLS[name].args_model.model_json_schema()
    defs, required = schema.get("$defs", {}), set(schema.get("required", []))
    params = []
    for arg, prop in schema.get("properties", {}).items():
        s = f"{arg}: {_type_of(prop, defs)}"
        if arg not in required:
            s += f" = {json.dumps(prop.get('default'))}"
        params.append(s)
    return f"{name}({', '.join(params)})"


def tool_catalog(agent: str) -> str:
    return "\n".join(f"- {tool_signature(t)}\n    {registry.TOOLS[t].description}" for t in AGENT_TOOLS[agent])


def build_messages(agent: str, question: str, plan: str, prompt_version: str | None = None) -> list[Message]:
    system = render("analyst", prompt_version, agent_name=agent, agent_description=AGENT_DESCRIPTIONS[agent],
                    tool_catalog=tool_catalog(agent), metrics=", ".join(NUMERIC_METRICS),
                    leagues=", ".join(LEAGUES), max_tool_calls=str(MAX_TOOL_CALLS))
    # The user's question goes in the user message, never into the system prompt.
    user = f"Question: {question}\nYour task from the router: {plan}"
    return [Message("system", system), Message("user", user)]


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------
class ToolRequest(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict, validation_alias=AliasChoices("arguments", "args"))


class AnalystDecision(BaseModel):
    tool_calls: list[ToolRequest] = Field(default_factory=list)
    reason: str = ""


def check_request(agent: str, req: ToolRequest) -> tuple[dict | None, dict | None]:
    """Return (validated_args, None) if the call may run, else (None, error)."""
    if req.tool not in registry.TOOLS:
        return None, {"code": "unknown_tool", "message": f"'{req.tool}' is not a tool.",
                      "allowed": AGENT_TOOLS[agent]}
    if req.tool not in AGENT_TOOLS[agent]:
        return None, {"code": "tool_not_allowed",
                      "message": f"The {agent} analyst may not use '{req.tool}'.", "allowed": AGENT_TOOLS[agent]}
    try:
        parsed = registry.TOOLS[req.tool].args_model(**req.arguments)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        return None, {"code": "invalid_arguments", "message": f"{req.tool}: {problems}",
                      "signature": tool_signature(req.tool)}
    return parsed.model_dump(mode="json", exclude_unset=True), None


def _repair_message(problem: str) -> Message:
    return Message("user", f"Your previous reply could not be used: {problem}\n"
                           "Reply again with JSON only, using only your listed tools and their exact argument names.")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def make_analyst(agent: str, llm: LLMClient, prompt_version: str | None = None,
                 data=None) -> Callable[[AgentState], dict]:
    """`data` lets tests inject a small Dataset; in production tools use the cached /datasets data."""
    if agent not in AGENT_TOOLS:
        raise ValueError(f"Unknown analyst '{agent}'")
    node_name = f"analyst:{agent}"

    def analyst(state: AgentState) -> dict:
        update: dict = {"tool_calls": [], "tool_results": [], "token_usage": [], "errors": []}
        if agent not in state.get("selected_agents", []):
            return update  # not selected by the router: do nothing

        def err(kind: str, message: str) -> None:
            update["errors"].append(AgentError(node=node_name, kind=kind, message=message[:500]))

        messages = build_messages(agent, state["question"], state.get("plan", {}).get(agent, state["question"]),
                                  prompt_version)
        approved: list[tuple[str, dict]] = []

        for attempt in range(MAX_REPAIRS + 1):
            try:
                resp = llm.complete(messages, purpose=node_name, json_mode=True, temperature=0.0)
            except LLMError as e:
                err("llm_error", f"{type(e).__name__}: {e}")
                return update
            update["token_usage"].append(usage_record(node_name, resp))

            try:
                raw = resp.json()
                if not isinstance(raw, dict):
                    raise ValueError("expected a JSON object")
                decision = AnalystDecision.model_validate(raw)
            except (LLMError, ValidationError, ValueError) as e:
                err("bad_output", f"{type(e).__name__}: {e}")
                problem = f"it was not valid JSON in the required shape ({type(e).__name__})."
            else:
                if not decision.tool_calls:
                    err("no_tool_calls", decision.reason or "Analyst selected no tools.")
                    return update
                rejected = []
                for req in decision.tool_calls[:MAX_TOOL_CALLS]:
                    args, problem_ = check_request(agent, req)
                    if problem_:
                        err(problem_["code"], problem_["message"])
                        rejected.append(problem_)
                    elif (req.tool, args) not in approved:  # skip exact duplicates
                        approved.append((req.tool, args))
                if len(decision.tool_calls) > MAX_TOOL_CALLS:
                    err("too_many_tool_calls", f"Only the first {MAX_TOOL_CALLS} tool calls were considered.")
                if approved:
                    break
                problem = "every tool call was rejected: " + json.dumps(rejected)
            if attempt < MAX_REPAIRS:
                messages = messages + [Message("assistant", resp.content), _repair_message(problem)]

        # Execute approved calls through the registry only.
        for i, (tool, args) in enumerate(approved, start=1):
            call_id = f"{agent}-{i}"
            update["tool_calls"].append(ToolCall(id=call_id, agent=agent, tool=tool, args=args))
            try:
                out = registry.call_tool(tool, args, data=data)
            except Exception as e:  # a bug inside a tool must not crash the graph
                out = {"ok": False, "error": {"code": "internal_error", "message": f"{type(e).__name__}: {e}",
                                              "suggestions": []}}
            update["tool_results"].append(ToolResult(
                call_id=call_id, agent=agent, tool=tool, args=args, ok=bool(out.get("ok")),
                result=out.get("result") if out.get("ok") else None,
                error=None if out.get("ok") else out.get("error")))
            if not out.get("ok"):
                e = out.get("error") or {}
                err("tool_error", f"{tool}: {e.get('code')}: {e.get('message')}")
        return update

    analyst.__name__ = f"analyst_{agent}"
    return analyst


def make_analysts(llm: LLMClient, prompt_version: str | None = None, data=None) -> dict[str, Callable]:
    """One node per specialist, keyed by agent name (used by the graph in Stage 7)."""
    return {a: make_analyst(a, llm, prompt_version, data) for a in AGENT_TOOLS}
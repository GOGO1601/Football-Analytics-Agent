"""
FastAPI service around the existing LangGraph workflow (agents/graph.py). It adds HTTP only:
validation, error responses, CORS, request ids and timing headers. No analysis happens here.

    GET  /health         is the process up? (touches nothing else)
    GET  /health/ready   are the compiled graph and the match data available? (no LLM, no tool)
    POST /api/ask        {"question": "..."} -> AskResponse (see app/schemas.py)
    GET  /docs           interactive OpenAPI docs

Run (from agent-backend/):
    uvicorn app.main:create_app --factory --reload --port 8000

One compiled graph is shared by all requests. That is safe because every request starts from
its own new_state(question), the graph has no checkpointer (so nothing carries over between
runs), and the offline LLM keeps no per-call state. HTTP status codes:
    200  the workflow finished: outcome is verified, no_evidence, fallback or unverified
    413  request body too large;  422  invalid request
    500  the workflow crashed or returned a result that breaks its own contract
Error bodies are RFC 9457 problem details (application/problem+json) and never contain a
traceback, a file path or exception text.
"""
from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from agents import graph as agent_graph
from agents.state import MAX_QUESTION_CHARS

from .config import Settings, get_settings
from .offline import MODEL_NAME as OFFLINE_MODEL
from .offline import offline_llm
from .schemas import AskRequest, AskResponse, IncompleteResult, Problem, to_response

API_VERSION = "0.1.0"
SERVICE = "football-analytics-api"
MAX_BODY_BYTES = 16_384          # a 1000-character question needs far less; larger bodies get 413
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

log = logging.getLogger("football_api")
if not log.handlers:                      # uvicorn only configures its own loggers
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ApiProblem(Exception):
    def __init__(self, status: int, title: str, detail: str):
        self.status, self.title, self.detail = status, title, detail


def problem(request: Request, status: int, title: str, detail: str, errors=None) -> JSONResponse:
    body = Problem(title=title, status=status, detail=detail, errors=errors,
                   request_id=getattr(request.state, "request_id", None))
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status,
                        media_type="application/problem+json")


def server_timing(trace: list, total_ms: float) -> str:
    """Server-Timing header: per-node time (summed when a node ran twice) plus the total."""
    per: dict[str, float] = {}
    for t in trace:
        per[t.node] = per.get(t.node, 0.0) + t.ms
    parts = [f"{node};dur={ms:.1f}" for node, ms in per.items()]
    return ", ".join(parts + [f"total;dur={total_ms:.1f}"])


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(settings: Settings | None = None, graph=None) -> FastAPI:
    """
    `graph` lets tests inject any object with the compiled graph's .invoke(); by default the
    real workflow is built with the offline LLM. Building calls no LLM and loads no data.
    """
    settings = settings or get_settings()
    llm_mode = "injected"
    if graph is None:
        if not settings.use_mock_llm:
            raise RuntimeError("USE_MOCK_LLM=false needs the real LLM client, which is added in a later "
                               "stage. Set USE_MOCK_LLM=true.")
        graph, llm_mode = agent_graph.build_graph(offline_llm()), OFFLINE_MODEL

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.preload_data:
            from tools.common import get_dataset
            get_dataset()                     # first question would otherwise wait ~1-2 s for the CSVs
        yield

    app = FastAPI(
        title="Football Analytics Agent API", version=API_VERSION, lifespan=lifespan,
        description="Ask questions about 30+ seasons of the top five European leagues. A LangGraph workflow "
                    "routes the question to specialist analysts, which run deterministic statistics tools; "
                    "the answer is written from those results and checked by a grounding guard. "
                    f"LLM mode: {llm_mode}.")
    app.state.graph, app.state.settings, app.state.llm_mode = graph, settings, llm_mode

    # -- middleware (the last one added is the outermost: CORS wraps everything) -------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            response = problem(request, 413, "Request too large",
                               f"Request bodies are limited to {MAX_BODY_BYTES} bytes.")
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        origin = request.headers.get("origin")
        if origin in settings.cors_origin_list:
            response.headers["Timing-Allow-Origin"] = origin      # lets the dashboard read Server-Timing
        return response

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Request-ID"],
                       expose_headers=["X-Request-ID", "Server-Timing"], max_age=600)

    # -- error handlers ---------------------------------------------------------------------
    @app.exception_handler(ApiProblem)
    async def _api_problem(request: Request, exc: ApiProblem):
        return problem(request, exc.status, exc.title, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError):
        errors = [{"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", ""), "type": e.get("type", "")}
                  for e in exc.errors()]                          # no echo of the submitted input
        return problem(request, 422, "Invalid request", "The request body did not pass validation.", errors)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return problem(request, exc.status_code, detail, detail)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.error("unhandled %s request_id=%s", type(exc).__name__, getattr(request.state, "request_id", "-"))
        return problem(request, 500, "Internal server error", "An unexpected error occurred.")

    # -- routes -------------------------------------------------------------------------------
    @app.get("/health", tags=["health"], summary="Process liveness (checks nothing else)")
    def health() -> dict:
        return {"status": "ok", "service": SERVICE, "version": API_VERSION}

    @app.get("/health/ready", tags=["health"], summary="Graph compiled and match data loaded (no LLM, no tool)",
             responses={503: {"model": Problem}})
    def ready(request: Request):
        checks: dict[str, str] = {}
        g = request.app.state.graph
        checks["graph"] = "ok" if callable(getattr(g, "invoke", None)) else "missing"
        try:
            from tools.common import get_dataset
            checks["dataset"] = f"ok ({len(get_dataset().matches)} matches)"
        except Exception as e:  # noqa: BLE001 - reported as not ready
            log.error("readiness: dataset unavailable (%s)", type(e).__name__)
            checks["dataset"] = "unavailable"
        checks["llm"] = f"{request.app.state.llm_mode} (not called by this check)"
        ok = checks["graph"] == "ok" and checks["dataset"].startswith("ok")
        body = {"status": "ready" if ok else "not_ready", "checks": checks}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/api/ask", response_model=AskResponse, tags=["analytics"],
              summary="Answer a football statistics question",
              responses={413: {"model": Problem}, 422: {"model": Problem}, 500: {"model": Problem}})
    def ask(body: AskRequest, request: Request, response: Response) -> AskResponse:
        rid = request.state.request_id
        t0 = perf_counter()
        try:
            final = agent_graph.ask(request.app.state.graph, body.question)
        except Exception as e:  # noqa: BLE001 - logged, then a 500 without internals
            log.error("ask failed request_id=%s error=%s", rid, type(e).__name__)
            raise ApiProblem(500, "Analysis failed", "The analysis could not be completed. Please try again.")
        try:
            out = to_response(final, request_id=rid, duration_ms=(perf_counter() - t0) * 1000)
        except (IncompleteResult, ValidationError, KeyError, TypeError) as e:
            log.error("ask returned an unusable result request_id=%s problem=%s: %s", rid, type(e).__name__,
                      str(e)[:200])
            raise ApiProblem(500, "Incomplete analysis result",
                             "The analysis finished without a usable result. Please try again.")
        response.headers["Server-Timing"] = server_timing(out.trace, out.duration_ms)
        log.info("ask request_id=%s outcome=%s retries=%d question_chars=%d ms=%.0f",
                 rid, out.outcome, out.retries, len(body.question), out.duration_ms)
        return out

    return app


__all__ = ["create_app", "API_VERSION", "MAX_QUESTION_CHARS"]
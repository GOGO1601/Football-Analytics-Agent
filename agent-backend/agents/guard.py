"""
Grounding guard: a deterministic check of the synthesiser's draft against the tool results
already stored in the state. No LLM call, no tool call, no recalculation of statistics.

    make_guard() -> node(state) -> {"grounding_result": GroundingResult}

Checks
  A/F  key_stats: the source is a successful tool result; the value appears in THAT result;
       and, where the label names a statistic ("p-value", "points", "win rate" ...), the value
       appears under a matching field of that result (so an r value cannot pose as a p-value).
  B    numbers in answer / evidence / explanation must match a number in the evidence.
       Only successful tool RESULTS are evidence (their fields and their own text such as
       interpretations and warnings). Tool arguments, the user's question and failed tools
       are CONTEXT: they supply team/league names and seasons, but their numbers never
       validate a claim. Two narrow exceptions, both checked by position not by value:
       a number that echoes the request ("top 5" when the question says "top 5", or the
       tool argument n=5) and a verbatim quotation of a failed tool's error message.
       Claims typed by their wording (p-value, r, R², confidence interval) must match evidence
       of the same type. Matching allows rounding to the precision written, proportion -> %
       conversion, scientific notation, sign-free magnitudes ("fell by 4.7 points") and
       inequalities ("p < 0.001"). Seasons, dates, significance levels ("5% level", "95% CI")
       and team/league names containing digits are not treated as statistics.
       Metric-aware (conservative): only when a number is followed by a metric word ("73 points",
       "26 wins", "21 games") it must match a value of that metric; and, if the clause names
       one team (or one season), a value from that team's (season's) row. Bare numbers are
       never attributed to a metric, team or season. A claim is only
       rejected when every matching value is positively identified as a DIFFERENT metric,
       team or season; values with no identifiable metric/team/season never cause rejection.
  C    significance wording must agree with the tool's own verdict (significant_at_0_05, the
       hypothesis tools' interpretation sentence, a trend's p-value). "Not significant" is
       recognised as a negative claim.
  D    causal wording ("causes", "leads to", "directly affects" ...) is rejected when the
       evidence is correlation/regression, unless it is negated.
  E    an incomplete-season warning must be acknowledged when the draft discusses that season
       or the tool covered only that season. A failed tool must be acknowledged.
  G    the result lists every issue (kind, field, claim, detail) plus feedback text in the
       format the synthesiser already reads on a retry.

This is a validator, not a proof of correctness. Known limitations:
  - metric/team/season checks only apply when the wording is explicit: "73 points" is checked
    as points, but "Arsenal's total was 73" is only checked against all result values; a
    clause that names two teams ("Chelsea beat Arsenal with 95 points") is not attributed
  - team names must be written as in the data ("Man United"); an alias in prose disables the
    team check for that clause (the metric check still applies)
  - a sentence can mislead without a wrong number ("collapsed", "weak link" for r = 0.597)
  - significance is checked per draft, not per sentence: with mixed verdicts in one answer,
    a wrong significance word cannot be attributed to the right test
  - years 1880-2035 written without a season format are treated as dates, not statistics
  - "significant" in its everyday sense is treated as a statistical claim (deliberately strict)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

from tools.common import LEAGUES

from .state import AgentError, AgentState, GroundingIssue, GroundingResult

NODE = "guard"
REL_TOL_COARSE = 0.05            # a 1-significant-figure claim must be within 5% of the value
NAME_KEYS = {"team", "teams", "team_a", "team_b", "league", "leagues"}

# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_NUM = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<sign>[+\-−])?"
    r"(?P<body>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:\s*[eE]\s*(?P<exp>[+\-−]?\d+)|\s*[x×]\s*10\s*\^?\s*(?P<exp2>[+\-−⁻]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+))?"
    r"(?P<ord>st|nd|rd|th)?"
    r"(?:(?P<pct>\s*%)|(?P<pp>\s*(?:pp\b|percentage[- ]points?)))?"
    r"(?![0-9])")

_MASKS = [
    re.compile(r"\b(?:18|19|20)\d{2}\s*[-/–]\s*(?:\d{4}|\d{2})\b"),            # seasons 2004-05, 2004/2005
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                                 # dates
    re.compile(r"\b(?:1|5|10)\s*%\s*(?:significance\s+)?level", re.I),           # 5% level
    re.compile(r"\b0?\.(?:01|05|1)\s+(?:significance\s+)?level", re.I),         # 0.05 level
    re.compile(r"(?:α|alpha)\s*(?:=|of|at)?\s*0?\.\d+", re.I),                  # alpha = 0.05
    re.compile(r"\b(?:90|95|99)\s*%(?=\s*(?:CI\b|C\.I\.|confidence))", re.I),  # 95% CI
]

_CTX_P = re.compile(r"(\bp[\s‑-]*values?|\bp)\s*(?:of|was|is|was only|=|<|>|≤|≥|<=|>=|:)?\s*(?:about|around|approximately|only)?\s*$", re.I)
_CTX_R2 = re.compile(r"(r\s*²|\br\s*\^?\s*2|r-squared|r squared|coefficient of determination)\s*(?:of|was|is|=|:)?\s*(?:about|around|approximately|only)?\s*$", re.I)
_CTX_R = re.compile(r"(\br|\brho|correlation(?: coefficient)?|pearson(?:'s)?(?: r)?|spearman(?:'s)?(?: rho| r)?)\s*(?:of|was|is|=|:)?\s*(?:about|around|approximately|only)?\s*$", re.I)
# "over"/"under" are deliberately absent: in football prose they usually mean "across" ("over 38 games")
_CTX_LT = re.compile(r"(<|≤|<=|less than|below|smaller than|fewer than|lower than)\s*$", re.I)
_CTX_GT = re.compile(r"(>|≥|>=|greater than|above|more than|exceeds?|exceeding|higher than)\s*$", re.I)
_CI_PHRASE = re.compile(r"confidence interval|\bC\.?I\.?\b", re.I)


@dataclass
class Claim:
    text: str
    value: float
    signed: bool               # sign written explicitly (or value written negative)
    half_unit: float           # half of the last written digit's place value
    sig: int                   # significant figures written
    pct: bool                  # "%" or "percentage points"
    kind: str                  # "p", "r", "r2", "ci" or "generic"
    ineq: str | None           # "<", ">" or None
    start: int
    integer: bool
    end: int = 0
    ordinal: bool = False


def _mask(text: str, names: Iterable[str]) -> str:
    chars = list(text)

    def blank(a: int, b: int) -> None:
        for i in range(a, b):
            chars[i] = " "
    for rx in _MASKS:
        for m in rx.finditer(text):
            blank(m.start(), m.end())
    low = text.lower()
    for n in names:
        if n and any(c.isdigit() for c in n):
            for m in re.finditer(re.escape(n.lower()), low):
                blank(m.start(), m.end())
    return "".join(chars)


def _sig_figs(body: str) -> int:
    digits = body.replace(",", "")
    if "." in digits:
        whole, frac = digits.split(".")
        s = (whole + frac).lstrip("0")
        return max(len(s), 1)
    s = digits.lstrip("0").rstrip("0")
    return max(len(s), 1)


def extract_claims(text: str, names: Iterable[str] = ()) -> list[Claim]:
    masked = _mask(text, names)
    claims: list[Claim] = []
    for m in _NUM.finditer(masked):
        body = m.group("body")
        sign = m.group("sign")
        prev = masked[m.start() - 1] if m.start() > 0 else ""
        # "trend-1", "offensive-2": an identifier, not a number
        if sign is None and prev in "-−" and m.start() > 1 and masked[m.start() - 2].isalnum():
            continue
        # "0.1-0.2" or "45%-50%": a dash right after a digit is a range, not a minus sign
        if sign == "-" and (prev.isdigit() or prev == "%"):
            sign = None
        exp_raw = m.group("exp") or m.group("exp2")
        exp = int(exp_raw.translate(_SUPER).replace("−", "-")) if exp_raw else 0
        mantissa = float(body.replace(",", ""))
        decimals = len(body.split(".")[1]) if "." in body else 0
        value = mantissa * (10 ** exp)
        if sign in ("-", "−"):
            value = -value
        half_unit = 0.5 * (10 ** (exp - decimals))
        before = text[max(0, m.start() - 40):m.start()]
        kind = "generic"
        if _CTX_P.search(before):
            kind = "p"
        elif _CTX_R2.search(before):
            kind = "r2"
        elif _CTX_R.search(before):
            kind = "r"
        ineq = "<" if _CTX_LT.search(before) else ">" if _CTX_GT.search(before) else None
        claims.append(Claim(text=text[m.start():m.end()].strip(), value=value, signed=bool(sign) or value < 0,
                            half_unit=half_unit, sig=_sig_figs(body), pct=bool(m.group("pct") or m.group("pp")),
                            kind=kind, ineq=ineq, start=m.start(), integer=decimals == 0 and exp == 0,
                            end=m.end(), ordinal=bool(m.group("ord"))))
    # the two numbers following "CI" / "confidence interval" are interval endpoints
    for ph in _CI_PHRASE.finditer(text):
        after = [c for c in claims if 0 <= c.start - ph.end() <= 60 and c.kind == "generic"]
        for c in after[:2]:
            c.kind = "ci"
    return claims


# ---------------------------------------------------------------------------
# Evidence index
# ---------------------------------------------------------------------------
@dataclass
class Ev:
    value: float
    kind: str
    key: str                    # normalised last key, e.g. "pvalue", "homewinrate", "interpretation"
    path: str
    call_id: str | None
    origin: str                 # "result" or "text" = evidence; "args" / "question" = context only
    team: str | None = None     # team of the row/result the number belongs to, if any
    season: str | None = None   # season of the row/result the number belongs to, if any


EVIDENCE_ORIGINS = ("result", "text")     # only these may validate a claim


# A metric word right after a number ("73 points") and the result keys that hold that metric.
METRIC_WORDS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(r"points per game|ppg\b", re.I), ("pointspergame",)),
    (re.compile(r"points?\b|pts\b", re.I), ("points",)),
    (re.compile(r"goals? per game", re.I), ("goalsforpergame", "goalsagainstpergame")),
    (re.compile(r"goals?\b", re.I), ("goals",)),
    (re.compile(r"wins?\b|victories\b", re.I), ("wins",)),
    (re.compile(r"draws?\b", re.I), ("draws",)),
    (re.compile(r"loss(?:es)?\b|defeats?\b", re.I), ("losses",)),
    (re.compile(r"shots on target", re.I), ("shotsontarget",)),
    (re.compile(r"shots?\b", re.I), ("shots",)),
    (re.compile(r"yellow cards?\b|yellows\b", re.I), ("yellow",)),
    (re.compile(r"red cards?\b|reds\b", re.I), ("red",)),
    (re.compile(r"fouls?\b", re.I), ("foul",)),
    (re.compile(r"matches\b|games\b|team-seasons\b", re.I), ("matches", "played", "n")),
]
ALL_METRIC_KEYS = [keys for _, keys in METRIC_WORDS]
_METRIC_LEAD = re.compile(r"^\s*(?:(?:more|fewer|extra|league|home|away|total)\s+)?", re.I)


def metric_after(text: str, end: int) -> tuple[str, ...] | None:
    """Result keys for a metric word that directly follows position `end`, e.g. '73 points' -> ('points',)."""
    rest = text[end:end + 40]
    rest = rest[_METRIC_LEAD.match(rest).end():]
    for rx, keys in METRIC_WORDS:
        if rx.match(rest):
            return keys
    return None


def _key_kind(path: list[str]) -> str:
    last = path[-1].lower() if path else ""
    if any("ci95" in p.lower() for p in path):
        return "ci"
    if last in ("p_value", "pvalue", "p"):
        return "p"
    if last in ("r_squared", "r2", "r2_label"):
        return "r2"
    if last == "r":
        return "r"
    return "generic"


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", k.lower())


@dataclass
class Evidence:
    numbers: list[Ev] = field(default_factory=list)
    strings: dict[str, list[str]] = field(default_factory=dict)   # call_id -> string leaves
    names: set[str] = field(default_factory=set)
    results: dict[str, dict] = field(default_factory=dict)        # call_id -> ToolResult
    verdicts: list[tuple[str, bool]] = field(default_factory=list)  # (call_id, significant?)
    observational: list[str] = field(default_factory=list)        # call_ids of correlation/regression
    quotes: set[str] = field(default_factory=set)                 # failed tools' error messages
    echoes: set[tuple[str, float]] = field(default_factory=set)   # (word before, value) from the request
    row_teams: set[str] = field(default_factory=set)              # teams that own result values
    row_teams_hint: set[str] = field(default_factory=set)         # team names seen in args/results

    @property
    def evidence(self) -> list[Ev]:
        return [e for e in self.numbers if e.origin in EVIDENCE_ORIGINS]

    def by_call(self, call_id: str) -> list[Ev]:
        return [e for e in self.numbers if e.call_id == call_id and e.origin in EVIDENCE_ORIGINS]


_SEASON_VALUE = re.compile(r"^(?:18|19|20)\d{2}-\d{2}$")


def _walk(obj: Any, path: list[str], call_id: str | None, ev: Evidence, origin: str,
          ctx: tuple[str | None, str | None] = (None, None)) -> None:
    """Index numbers. ctx = (team, season) of the enclosing row/result, inherited downwards."""
    team, season = ctx
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return
        ev.numbers.append(Ev(float(obj), _key_kind(path), _norm_key(path[-1]) if path else "",
                             ".".join(path), call_id, origin, team, season))
        return
    if isinstance(obj, str):
        if call_id:
            ev.strings.setdefault(call_id, []).append(obj)
        if path and path[-1].lower() in NAME_KEYS or (len(path) >= 2 and path[-2].lower() in NAME_KEYS):
            ev.names.add(obj)
        if re.fullmatch(r"\s*[+\-−]?\d+(?:\.\d+)?\s*", obj):        # numeric strings such as r2_label "0.356"
            ev.numbers.append(Ev(float(obj.replace("−", "-")), _key_kind(path), _norm_key(path[-1]) if path else "",
                                 ".".join(path), call_id, origin, team, season))
        elif any(c.isdigit() for c in obj):
            text_origin = "text" if origin == "result" else origin
            for c in extract_claims(obj, ev.names | set(LEAGUES)):
                kind = _key_kind(path) if _key_kind(path) != "generic" else c.kind
                # a number in tool text followed by a metric word ("380 matches") carries that metric
                keys = metric_after(obj, c.end)
                key = keys[0] if keys else (_norm_key(path[-1]) if path else "")
                ev.numbers.append(Ev(c.value, kind, key, ".".join(path), call_id, text_origin))
        return
    if isinstance(obj, dict):
        t = obj.get("Team", obj.get("team"))
        sn = obj.get("Season", obj.get("season"))
        if isinstance(t, str):
            team = t
        if isinstance(sn, str) and _SEASON_VALUE.match(sn):
            season = sn
        for k, v in obj.items():
            child_team = k if isinstance(v, dict) and k in ev.row_teams_hint else team  # team_comparison summary
            _walk(v, path + [str(k)], call_id, ev, origin, (child_team, season))
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:                      # list items keep the parent key, e.g. ci95 -> both bounds are "ci"
            _walk(v, path, call_id, ev, origin, (team, season))


_VERDICT_NEG = re.compile(r"\b(?:not statistically significant|not significant)\b", re.I)
_VERDICT_POS = re.compile(r"\b(?:statistically significant at|significant at the 5% level)", re.I)


def _verdict(tool: str, res: dict) -> bool | None:
    """The tool's own significance verdict, or None if the tool does not give one."""
    if not isinstance(res, dict):
        return None
    if isinstance(res.get("significant_at_0_05"), bool):
        return res["significant_at_0_05"]
    interp = res.get("interpretation") or ""
    if tool in ("home_advantage_test", "hypothesis_test") and interp:
        if _VERDICT_NEG.search(interp):
            return False
        if _VERDICT_POS.search(interp):
            return True
    if tool == "regression_analysis" and isinstance(res.get("p_value"), (int, float)):
        return res["p_value"] < 0.05                         # same 5% convention the tools use
    trend = res.get("linear_trend")
    if tool == "season_trend" and isinstance(trend, dict) and isinstance(trend.get("p_value"), (int, float)):
        return trend["p_value"] < 0.05
    return None


def build_evidence(state: AgentState) -> Evidence:
    ev = Evidence()
    ev.names |= set(LEAGUES)
    results = state.get("tool_results", [])
    # first pass: names, so digits in team names are masked when parsing tool text
    for r in results:
        for v in (r.get("args") or {}).values():
            if isinstance(v, str):
                ev.names.add(v)
            elif isinstance(v, list):
                ev.names |= {x for x in v if isinstance(x, str)}
        if r["ok"] and isinstance(r.get("result"), dict):
            _collect_names(r["result"], [], ev.names)
    ev.row_teams_hint = {n for n in ev.names if n not in LEAGUES}
    for r in results:
        cid = r["call_id"]
        ev.results[cid] = r
        args = r.get("args") or {}
        _walk(args, ["args"], cid, ev, "args")                      # context only
        if isinstance(args.get("n"), int):
            for w in ("top", "bottom", "first", "last", "best", "worst"):
                ev.echoes.add((w, float(args["n"])))
        if r["ok"]:
            _walk(r.get("result") or {}, [], cid, ev, "result")
            v = _verdict(r["tool"], r.get("result") or {})
            if v is not None:
                ev.verdicts.append((cid, v))
            if r["tool"] in ("correlation_analysis", "regression_analysis"):
                ev.observational.append(cid)
        else:
            msg = (r.get("error") or {}).get("message", "")
            if msg:
                ev.quotes.add(msg)                                  # quotable, never evidence
    question = state.get("question", "")
    for c in extract_claims(question, ev.names):                    # context only
        ev.numbers.append(Ev(c.value, "generic", "question", "question", None, "question"))
        ev.echoes.add((_word_before(question, c.start), c.value))
    ev.row_teams = {e.team for e in ev.evidence if e.team}
    return ev


def _word_before(text: str, start: int) -> str:
    m = re.search(r"([A-Za-z]+)[^A-Za-z0-9]*$", text[:start])
    return m.group(1).lower() if m else ""


def _collect_names(obj: Any, path: list[str], names: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_names(v, path + [str(k)], names)
    elif isinstance(obj, list):
        for v in obj:
            _collect_names(v, path, names)
    elif isinstance(obj, str) and path and path[-1].lower() in NAME_KEYS:
        names.add(obj)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _close(c: Claim, x: float) -> bool:
    d = abs(c.value - x)
    if d > c.half_unit * (1 + 1e-9) + 1e-12 * max(1.0, abs(c.value)):
        return False
    if c.sig <= 1 and d > 0 and x != 0 and d / abs(x) > REL_TOL_COARSE:
        return False
    return True


def _scales(c: Claim) -> tuple[float, ...]:
    return (1.0, 100.0) if (c.pct or c.kind == "ci") else (1.0,)


def match(c: Claim, candidates: Iterable[Ev]) -> tuple[Ev | None, bool]:
    """Return (matching evidence, matched_only_by_magnitude)."""
    by_abs = None
    for e in candidates:
        for s in _scales(c):
            x = e.value * s
            if c.ineq == "<":
                if x < c.value:
                    return e, False
                continue
            if c.ineq == ">":
                if x > c.value:
                    return e, False
                continue
            if _close(c, x):
                return e, False
            if not c.signed and x < 0 and _close(c, -x) and by_abs is None:
                by_abs = e
    return (by_abs, True) if by_abs else (None, False)


def _candidates(c: Claim, ev: Evidence) -> list[Ev]:
    """Only successful results are evidence. Typed claims additionally need evidence of their type."""
    if c.kind == "generic":
        return ev.evidence
    return [e for e in ev.evidence if e.kind == c.kind]


def all_matches(c: Claim, candidates: Iterable[Ev]) -> list[tuple[Ev, bool]]:
    """Every (evidence, matched_only_by_magnitude) that supports the claim's value."""
    out = []
    for e in candidates:
        hit, by_abs = match(c, [e])
        if hit is not None:
            out.append((e, by_abs))
    return out


def _closest(c: Claim, pool: list[Ev], k: int = 2) -> str:
    vals = sorted(pool, key=lambda e: (min(abs(e.value * s - abs(c.value)) for s in _scales(c)),
                                       e.origin != "result"))[:k]
    return "; ".join(f"{e.value:.6g} ({e.call_id or e.origin}: {e.path})" for e in vals)


def _year_like(c: Claim) -> bool:
    return c.integer and not c.pct and c.kind == "generic" and 1880 <= abs(c.value) <= 2035


# ---------------------------------------------------------------------------
# Text checks
# ---------------------------------------------------------------------------
_SIG = re.compile(r"\b(?:in|non-?)?significan(?:t|tly)\b", re.I)
_NEGATOR = re.compile(r"\b(?:not|no|nor|never|without|neither|cannot|isn't|wasn't|aren't|weren't|didn't|doesn't|"
                      r"don't|fails? to|failed to|lack of)\b[^.;:!?]{0,25}$", re.I)
_CAUSAL = re.compile(r"\b(?:caus(?:es|ed|ing|e)|causal(?:ly)?|leads? to|led to|drives|driving|produces|"
                     r"is responsible for|are responsible for|directly (?:affects?|impacts?|influences?|increases?|"
                     r"reduces?|improves?|determines?)|boosts?|guarantees?|makes teams (?:win|earn|score)|"
                     r"will (?:earn|win|gain|add|increase|result)|(?:has|have|had) an? (?:effect|impact|influence) on)\b",
                     re.I)
_CAUSAL_NEG = re.compile(r"\b(?:not|n't|no|cannot|never|without|rather than|whether|nor|not necessarily)\b[^.;:]{0,40}$",
                         re.I)
_RISE = re.compile(r"\b(?:rose|rise|risen|increased?|higher|gain(?:ed)?|grew|up by|improved?)\b", re.I)
_FALL = re.compile(r"\b(?:fell|fall|fallen|decreased?|lower|drop(?:ped)?|declined?|reduced?|down by|worse)\b", re.I)
_INCOMPLETE = re.compile(r"^(?P<league>.+?) (?P<season>\d{4}-\d{2}) is incomplete")
_ACK_INCOMPLETE = re.compile(r"incomplete|partial|not complete|missing (?:matches|games|data)|only \d+ of \d+|"
                             r"\d+ of \d+ matches|fewer matches|unfinished|in progress|ongoing|not finished|part of the season",
                             re.I)
_ACK_FAILURE = re.compile(r"could not|couldn't|failed|unknown|not found|unavailable|error|no data|not available|"
                          r"insufficient|unable|missing", re.I)
SINGLE_SEASON_ARGS = ("season", "season_from", "season_to")


_RANGE_WORD = re.compile(r"^\s*(?:to|until|through|–|-|—)\s*(?:18|19|20)\d{2}", re.I)
_RANGE_BEFORE = re.compile(r"(?:(?:18|19|20)\d{2}-\d{2})\s*(?:to|until|through|–|—|-)\s*$", re.I)


def _discussed_alone(season: str, text: str) -> bool:
    """True if the season is mentioned on its own, not only as the start or end of a range."""
    for m in re.finditer(re.escape(season), text):
        after, before = text[m.end():m.end() + 12], text[max(0, m.start() - 14):m.start()]
        if not _RANGE_WORD.search(after) and not _RANGE_BEFORE.search(before):
            return True
    return False


def _text_fields(draft: dict) -> list[tuple[str, str]]:
    out = [("answer", draft.get("answer") or "")]
    out += [(f"evidence[{i}]", t) for i, t in enumerate(draft.get("evidence") or []) if isinstance(t, str)]
    out.append(("explanation", draft.get("explanation") or ""))
    return out


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
LABEL_RULES: list[tuple[re.Pattern, str | None, tuple[str, ...]]] = [
    (re.compile(r"r²|\br2\b|r-squared|r squared|coefficient of determination", re.I), "r2", ()),
    (re.compile(r"\bp[- ]?values?\b|\bp\b", re.I), "p", ()),
    (re.compile(r"correlation|pearson|spearman|\brho\b|\br\b", re.I), "r", ()),
    (re.compile(r"confidence interval|\bci\b", re.I), "ci", ()),
    (re.compile(r"cohen|effect size|hedges", re.I), None, ("cohens", "hedges")),
    (re.compile(r"slope|per unit|coefficient", re.I), None, ("slope",)),
    (re.compile(r"change|difference|gap", re.I), None, ("difference", "change")),
    (re.compile(r"win rate|win %|win percentage|win pct", re.I), None, ("winrate", "winpct")),
    (re.compile(r"points per game|\bppg\b", re.I), None, ("pointspergame",)),
    (re.compile(r"goal margin|goal difference|goaldiff|\bgd\b", re.I), None, ("goaldiff", "goalmargin")),
    (re.compile(r"points", re.I), None, ("points",)),
    (re.compile(r"shots on target", re.I), None, ("shotsontarget",)),
    (re.compile(r"shots", re.I), None, ("shots",)),
    (re.compile(r"yellow", re.I), None, ("yellow",)),
    (re.compile(r"red card", re.I), None, ("red",)),
    (re.compile(r"foul", re.I), None, ("foul",)),
    (re.compile(r"goals", re.I), None, ("goals",)),
    (re.compile(r"\bwins?\b|victories", re.I), None, ("wins",)),
    (re.compile(r"\bdraws?\b", re.I), None, ("draws",)),
    (re.compile(r"loss|losses|defeats", re.I), None, ("losses",)),
    (re.compile(r"matches|games|sample size|\bn\b|team-seasons", re.I), None, ("matches", "played", "n")),
    (re.compile(r"position|rank|place", re.I), None, ("position",)),
]


def _label_rule(label: str) -> tuple[str | None, tuple[str, ...]] | None:
    for rx, kind, keys in LABEL_RULES:
        if rx.search(label):
            return kind, keys
    return None


def _key_matches(e: Ev, keys: tuple[str, ...]) -> bool:
    return any(e.key == k or (len(k) > 1 and k in e.key) for k in keys)


# ---------------------------------------------------------------------------
# Context-aware checks (metric / team / season)
# ---------------------------------------------------------------------------
_CLAUSE_SPLIT = re.compile(r"[,;:()\[\]]| — | - ")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")
_SEASON_IN_TEXT = re.compile(r"\b((?:18|19|20)\d{2})\s*[-/–]\s*(\d{2}|\d{4})\b")
_AGGREGATE = re.compile(r"\b(?:average[ds]?|averaging|mean|total|overall|combined|across|in all|per season|career|"
                        r"since|from|between|over the|seasons)\b", re.I)


def _span(text: str, pos: int, splitter: re.Pattern) -> str:
    starts = [0] + [m.end() for m in splitter.finditer(text) if m.end() <= pos]
    ends = [m.start() for m in splitter.finditer(text) if m.start() >= pos] + [len(text)]
    return text[max(starts):min(ends)]


def _teams_in(segment: str, teams: set[str]) -> set[str]:
    return {t for t in teams if re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", segment, re.I)}


def _seasons_in(segment: str) -> set[str]:
    out = set()
    for m in _SEASON_IN_TEXT.finditer(segment):
        if not _discussed_alone(m.group(0), segment):
            continue
        out.add(f"{m.group(1)}-{m.group(2)[-2:]}")
    return out


def _one(clause_items: set[str], sentence_items: set[str]) -> str | None:
    if len(clause_items) == 1:
        return next(iter(clause_items))
    if not clause_items and len(sentence_items) == 1:
        return next(iter(sentence_items))
    return None


def _is_echo(c: Claim, text: str, ev: Evidence) -> bool:
    """A number repeating the request ('top 5'), not a statistic: same preceding word, no metric word after."""
    if c.kind != "generic" or c.pct or metric_after(text, c.end):
        return False
    return (_word_before(text, c.start), c.value) in ev.echoes


def _neutral(e: Ev) -> bool:
    return not any(_key_matches(e, keys) for keys in ALL_METRIC_KEYS)


def _context_problem(c: Claim, text: str, ev: Evidence, matches: list[tuple[Ev, bool]]) -> tuple[str, str] | None:
    """
    Conservative metric / team / season check. Returns (issue kind, detail) only when EVERY value
    that matches the number is positively identified as a different metric, team or season.
    """
    if c.kind != "generic" or c.pct or c.ineq:
        return None
    want = metric_after(text, c.end)
    if want is None:              # no metric word: never attribute a bare number to a metric, team or season
        return None
    sentence = _span(text, c.start, _SENTENCE_SPLIT)
    clause = _span(text, c.start, _CLAUSE_SPLIT)
    team = _one(_teams_in(clause, ev.row_teams), _teams_in(sentence, ev.row_teams))
    season = None
    if not _AGGREGATE.search(sentence):
        season = _one(_seasons_in(clause), _seasons_in(sentence))

    def metric_ok(e: Ev) -> bool:
        return _neutral(e) or _key_matches(e, want)

    def team_ok(e: Ev) -> bool:
        return team is None or e.team is None or e.team.lower() == team.lower()

    def season_ok(e: Ev) -> bool:
        return season is None or e.season is None or e.season == season

    evs = [e for e, _ in matches]
    if any(metric_ok(e) and team_ok(e) and season_ok(e) for e in evs):
        return None
    where = "; ".join(f"{e.call_id}: {e.path}" + (f" ({e.team}" + (f", {e.season}" if e.season else "") + ")"
                                                  if e.team else "") for e in evs[:3])
    if not any(metric_ok(e) for e in evs):
        right = [e for e in ev.evidence if _key_matches(e, want) and team_ok(e) and season_ok(e)]
        hint = f"; the {want[0]} value(s) in the evidence: {_closest(c, right)}" if right else ""
        return "metric_mismatch", (f"'{c.text}' is described as {want[0]}, but it matches {where}, "
                                   f"which is a different statistic{hint}")
    if not any(metric_ok(e) and team_ok(e) for e in evs):
        right = [e for e in ev.evidence if _key_matches(e, want) and e.team and e.team.lower() == team.lower()]
        hint = f"; {team}'s value(s): {_closest(c, right)}" if right else ""
        return "entity_mismatch", f"'{c.text}' is attributed to {team}, but it belongs to {where}{hint}"
    right = [e for e in ev.evidence if _key_matches(e, want) and team_ok(e) and e.season == season]
    hint = f"; the {season} value(s): {_closest(c, right)}" if right else ""
    return "season_mismatch", f"'{c.text}' is attributed to {season}, but it belongs to {where}{hint}"


def check(state: AgentState) -> GroundingResult:
    issues: list[GroundingIssue] = []
    checked = 0

    def issue(kind: str, fld: str, claim: str, detail: str) -> None:
        issues.append(GroundingIssue(kind=kind, field=fld, claim=claim, detail=detail))

    draft = state.get("draft_answer")
    if not draft or not (draft.get("answer") or "").strip():
        issue("empty_draft", "answer", "", "There is no draft answer to check.")
        return _result(issues, checked)

    ev = build_evidence(state)
    successful = [cid for cid, r in ev.results.items() if r["ok"]]
    fields = _text_fields(draft)
    all_text = "\n".join(t for _, t in fields)

    # ---- B: numbers in the prose -------------------------------------------------
    for fld, text in fields:
        # verbatim quotes of failed tools' error messages are reported, not claimed
        for c in extract_claims(text, ev.names | ev.quotes):
            cands = _candidates(c, ev)
            matches = all_matches(c, cands)
            if not matches:
                if _year_like(c) or _is_echo(c, text, ev):
                    continue
                checked += 1
                what = {"p": "p-value", "r": "correlation (r)", "r2": "R²", "ci": "confidence-interval bound"}.get(c.kind)
                if what and not cands:
                    issue("unsupported_number", fld, c.text, f"no {what} appears in the tool results")
                else:
                    label = f"{what} " if what else ""
                    issue("unsupported_number", fld, c.text,
                          f"{label}'{c.text}' does not match the tool results; closest values: "
                          f"{_closest(c, cands or ev.evidence)}")
                continue
            checked += 1
            problem = _context_problem(c, text, ev, matches)
            if problem:
                issue(problem[0], fld, c.text, problem[1])
                continue
            hit, by_abs = next((m for m in matches if not m[1]), matches[0])   # prefer an exact-sign match
            if by_abs:
                window = text[max(0, c.start - 60):c.start]
                if _RISE.search(window) and not _FALL.search(window):
                    issue("direction_contradiction", fld, c.text,
                          f"'{c.text}' matches a negative value ({hit.value:.6g} at {hit.call_id}: {hit.path}) "
                          f"but is described as an increase")

    # ---- A/F: key stats ----------------------------------------------------------------
    for i, ks in enumerate(draft.get("key_stats") or []):
        fld = f"key_stats[{i}]"
        checked += 1
        src, value, label = ks.get("source", ""), str(ks.get("value", "")), ks.get("label", "")
        r = ev.results.get(src)
        if r is None:
            issue("invalid_key_stat_source", fld, src, f"no tool result has id '{src}'; valid ids: {sorted(successful)}")
            continue
        if not r["ok"]:
            issue("invalid_key_stat_source", fld, src, f"'{src}' is a failed tool call ({r['tool']}); it is not evidence")
            continue
        claims = extract_claims(value, ev.names)
        if not claims:
            if value.strip().lower() not in " ".join(ev.strings.get(src, [])).lower():
                issue("key_stat_value_unsupported", fld, value, f"'{value}' does not appear in {src}")
            continue
        rule = _label_rule(label)            # (kind, keys) named by the label, or None
        pool = ev.by_call(src)                # numbers from THIS source only
        for c in claims:
            if rule is None:
                typed = pool
            elif rule[0]:                     # p / r / r2 / ci: match evidence of that kind
                c = replace(c, kind=rule[0])
                typed = [e for e in pool if e.kind == rule[0]]
            else:                             # e.g. "points": match fields whose key says points
                typed = [e for e in pool if _key_matches(e, rule[1])]
            hit, _ = match(c, typed)
            if hit:
                continue
            any_hit, _ = match(replace(c, kind="generic"), pool)
            if any_hit is not None and rule:
                issue("key_stat_label_mismatch", fld, f"{label}: {value}",
                      f"'{c.text}' appears in {src} as {any_hit.path}, which is not the statistic named by "
                      f"the label '{label}'" + (f"; that statistic is {_closest(c, typed, 1)}" if typed else ""))
            else:
                issue("key_stat_value_unsupported", fld, f"{label}: {value}",
                      f"'{c.text}' is not a value in {src}" + (f"; closest: {_closest(c, typed or pool)}" if pool else ""))

    # ---- C: significance ---------------------------------------------------------------
    verdicts = [v for _, v in ev.verdicts]
    for fld, text in fields:
        for m in _SIG.finditer(text):
            word = m.group(0).lower()
            before = text[max(0, m.start() - 40):m.start()]
            negative = word.startswith(("in", "non")) or bool(_NEGATOR.search(before))
            checked += 1
            snippet = text[max(0, m.start() - 30):m.end() + 20].strip()
            if not negative:
                if not verdicts:
                    issue("significance_without_test", fld, snippet,
                          "the draft claims statistical significance but no significance test was run")
                elif not any(verdicts):
                    issue("significance_contradicted", fld, snippet,
                          "the draft calls a result significant but every test in the evidence says it is not "
                          f"significant ({_verdict_summary(ev)})")
            elif verdicts and all(verdicts):
                issue("significance_contradicted", fld, snippet,
                      f"the draft says not significant but the test(s) say significant ({_verdict_summary(ev)})")

    # ---- D: causal wording on observational results ------------------------------------
    if ev.observational:
        for fld, text in fields:
            for m in _CAUSAL.finditer(text):
                checked += 1
                if _CAUSAL_NEG.search(text[max(0, m.start() - 45):m.start()]):
                    continue
                issue("unsupported_causal_claim", fld, m.group(0),
                      f"'{m.group(0)}' states causation, but {', '.join(ev.observational)} is a correlation/regression "
                      "result, which only shows an association")

    # ---- E: caveats ----------------------------------------------------------------------
    for cid in successful:
        r = ev.results[cid]
        for w in (r.get("result") or {}).get("warnings") or []:
            m = _INCOMPLETE.match(w)
            if not m:
                continue
            season = m.group("season")
            single = any(str((r.get("args") or {}).get(a, "")).replace("/", "-")[:7] == season
                         for a in SINGLE_SEASON_ARGS)
            if single or _discussed_alone(season, all_text):
                checked += 1
                if not _ACK_INCOMPLETE.search(all_text):
                    issue("missing_caveat", "answer", season,
                          f"the answer relies on {season} but does not mention the tool warning: '{w}'")
    failed = [r for r in ev.results.values() if not r["ok"]]
    if failed and successful:
        checked += 1
        if not _ACK_FAILURE.search(all_text):
            names = ", ".join(f"{r['tool']} ({r['call_id']})" for r in failed)
            issue("missing_failure_note", "answer", names,
                  f"{names} failed but the draft does not say that part of the analysis could not be completed")
    return _result(issues, checked)


def _verdict_summary(ev: Evidence) -> str:
    return ", ".join(f"{cid}: {'significant' if v else 'not significant'}" for cid, v in ev.verdicts)


def _result(issues: list[GroundingIssue], checked: int) -> GroundingResult:
    if not issues:
        return GroundingResult(passed=True, checked=checked, unsupported=[], feedback="", issues=[])
    lines = [f"- [{i['kind']}] {i['field']}: {i['detail']}" for i in issues]
    return GroundingResult(passed=False, checked=checked,
                           unsupported=list(dict.fromkeys(i["claim"] for i in issues if i["claim"])),
                           feedback="Fix these problems:\n" + "\n".join(lines), issues=issues)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def make_guard() -> Callable[[AgentState], dict]:
    """The guard takes no LLM: it is pure Python over the state."""
    def guard(state: AgentState) -> dict:
        try:
            return {"grounding_result": check(state)}
        except Exception as e:  # fail closed: an unverifiable draft is not accepted
            issue = GroundingIssue(kind="guard_error", field="", claim="",
                                   detail=f"the guard could not check this draft ({type(e).__name__}: {e})")
            return {"grounding_result": GroundingResult(passed=False, checked=0, unsupported=[],
                                                        feedback=f"- [guard_error] {issue['detail']}", issues=[issue]),
                    "errors": [AgentError(node=NODE, kind="guard_error", message=f"{type(e).__name__}: {e}")]}
    return guard
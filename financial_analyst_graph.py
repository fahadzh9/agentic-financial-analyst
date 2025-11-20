import os

# 🔑 Keys (set via env vars)
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# 🤖 Models
os.environ["CLAUDE_MODEL"] = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
os.environ["STRATEGY_MODEL"] = os.getenv("STRATEGY_MODEL", "claude-sonnet-4-5-20250929")

# ⚙️ Feature toggles
USE_SQL_LLM = True
ENABLE_TRACE = True

# 🌍 Macro mode
MACRO_MODE = os.getenv("MACRO_MODE", "llm")  # "llm" or "static"
STATIC_MACRO_QUERY = "Saudi Arabia macro consumer demand inflation retail confidence policy latest"

import os, time, anthropic, httpx
from anthropic import Anthropic

import psycopg
from psycopg.rows import dict_row

# 🛠️ DB creds from environment (for Supabase/Postgres, etc.)
PG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
    "schema": os.getenv("PGSCHEMA", "public"),
}

ALLOWED_TABLES = [
    "saudimarketcompanies",
    "balancesheets",
    "incomestatements",
    "sectors",
    "clients",
    "clientcompanies",
    "clientbalancesheets",
    "clientincomestatements",
]


def db_session():
    return psycopg.connect(
        host=PG["host"],
        port=PG["port"],
        dbname=PG["dbname"],
        user=PG["user"],
        password=PG["password"],
    )


from typing import Dict, List


def introspect_columns():
    q = (
        "SELECT table_name, column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_schema=%(schema)s AND table_name=ANY(%(tables)s) "
        "ORDER BY table_name, ordinal_position;"
    )
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(q, {"schema": PG["schema"], "tables": ALLOWED_TABLES})
            return cur.fetchall()


meta = introspect_columns()
by_table_cols: Dict[str, List[str]] = {}
for r in meta:
    by_table_cols.setdefault(r["table_name"], []).append(r["column_name"])

MARKET_TABLES = [
    "saudimarketcompanies",
    "balancesheets",
    "incomestatements",
    "sectors",
]
CLIENT_TABLES = [
    "clients",
    "clientcompanies",
    "clientbalancesheets",
    "clientincomestatements",
]


def pick(cols: List[str], cands: List[str]):
    lc = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def infer_client_roles(by: Dict[str, List[str]]):
    # choose income table first
    t = (
        "clientincomestatements"
        if "clientincomestatements" in by
        else next((x for x in by if "income" in x), "clientincomestatements")
    )
    cols = by.get(t, [])
    comp = by.get("clientcompanies", [])
    return {
        "table": t,
        "company_id": pick(cols, ["client_company_id"]) or "client_company_id",
        "company_name": pick(comp, ["company_name", "name"]) or "company_name",
        "year": pick(cols, ["year"]) or "year",
        "quarter": pick(cols, ["quarter", "qtr"]) or "quarter",
        "revenue": pick(cols, ["total_revenue", "revenue", "net_sales", "sales"]) or "total_revenue",
        "op_inc": pick(
            cols,
            [
                "operating_income",
                "operating_profit",
                "operating_income_loss",
                "total_operating_income",
                "total_operating_income_as_reported",
            ],
        )
        or "operating_income",
    }


def infer_market_roles(by: Dict[str, List[str]]):
    t = (
        "incomestatements"
        if "incomestatements" in by
        else next((x for x in by if "income" in x), "incomestatements")
    )
    cols = by.get(t, [])
    comp = by.get("saudimarketcompanies", [])
    return {
        "table": t,
        "company_id": pick(cols, ["company_id"]) or "company_id",
        "ticker": pick(comp, ["ticker"]) or "ticker",
        "company_name": pick(comp, ["company_name", "name"]) or "company_name",
        "year": pick(cols, ["year"]) or "year",
        "quarter": pick(cols, ["quarter", "qtr"]) or "quarter",
        "revenue": pick(cols, ["total_revenue", "revenue", "net_sales", "sales"]) or "total_revenue",
        "op_inc": pick(
            cols,
            [
                "operating_income",
                "operating_profit",
                "operating_income_loss",
                "total_operating_income",
                "total_operating_income_as_reported",
            ],
        )
        or "operating_income",
    }


ROLES = {"client": infer_client_roles(by_table_cols), "market": infer_market_roles(by_table_cols)}


def describe_schema():
    from textwrap import indent

    lines = []
    lines.append("Tables and columns:")
    order = [
        t for t in MARKET_TABLES if t in by_table_cols
    ] + [t for t in CLIENT_TABLES if t in by_table_cols] + [
        t for t in by_table_cols if t not in MARKET_TABLES + CLIENT_TABLES
    ]
    for t in order:
        lines.append(f"- {t}(" + ", ".join(by_table_cols[t]) + ")")
    lines.append("")
    lines.append("Roles:")
    for k, v in ROLES.items():
        lines.append(f"- {k}:")
        for kk, vv in v.items():
            lines.append(f"    {kk}: {vv}")
    return "\n".join(lines)


import re, json, requests
from typing import Dict, Any


def tavily_search(query: str, max_results: int = 4) -> Dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return {"error": "TAVILY_API_KEY not set", "results": []}
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_domains": [],
        "exclude_domains": [],
        "include_answer": True,
        "include_raw_content": False,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data


TRACE_LAST: Dict[str, Any] = {}


def reset_trace():
    global TRACE_LAST
    TRACE_LAST = {}


def get_trace():
    return TRACE_LAST


def sql_llm_agent(user_query: str, client_company_id: int | str) -> Dict[str, Any]:
    """
    Use Anthropic to synthesize a safe SQL query for the client's data,
    along with parameters, and a natural-language rationale.
    """
    global TRACE_LAST
    aclient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    schema_text = describe_schema()
    prompt = f"""
You are a senior financial data engineer. We have the following PostgreSQL schema and roles:

{schema_text}

The user query is:

{user_query!r}

The caller will supply a `client_company_id` = {client_company_id!r}.

Task:
1. Generate a single, safe SQL query that:
   - Filters on the given client_company_id for client data.
   - Joins to market tables when relevant.
   - Returns a tidy time series of quarterly metrics to support the question (e.g. revenue, operating income, margins, etc.).
2. Use **named parameters** in PostgreSQL format (e.g. `%(client_company_id)s`) instead of positional markers like `$1`.
   - Always include a `params` JSON object whose keys match the named parameters you used.
   - Do not emit `$1`, `$2`, etc.
3. Provide a short explanation of what you are doing.

Output strictly as JSON with keys:
- "sql": string
- "params": object
- "rationale": string
"""
    t0 = time.perf_counter()
    resp = aclient.messages.create(
        model=os.getenv("CLAUDE_MODEL"),
        max_tokens=800,
        temperature=0,
        system="You only output valid JSON for a financial SQL generation task.",
        messages=[{"role": "user", "content": prompt}],
    )
    dt = int((time.perf_counter() - t0) * 1000)
    raw = "".join(
        [b.text for b in resp.content if getattr(b, "type", None) == "text"]  # type: ignore[attr-defined]
    ).strip()
    try:
        data = json.loads(raw)
    except Exception:
        # Try to extract JSON block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    TRACE_LAST["sql_agent"] = {
        "provider": "anthropic",
        "model": resp.model,
        "latency_ms": dt,
        "rationale": data.get("rationale"),
        "json_fragment": raw[:1000],
    }
    return data


from decimal import Decimal
import datetime as _dt
import numpy as _np


def run_safe_query(sql: str, params: Dict[str, Any] | None = None):
    """
    Guardrails: block destructive SQL; only allow SELECT.
    """
    up = sql.strip().upper()
    if any(
        up.startswith(x)
        for x in ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "TRUNCATE ", "CREATE "]
    ):
        raise ValueError("Unsafe SQL blocked by guardrails")

    def normalize_sql_params(raw_sql: str, raw_params: Dict[str, Any] | None):
        """Translate positional placeholders ($1) into psycopg-friendly params."""

        if not raw_params:
            return raw_sql, raw_params
        if isinstance(raw_params, dict) and re.search(r"\$\d+", raw_sql):
            keys = list(raw_params.keys())

            def value_for_index(idx: int):
                # Prefer matching key order, otherwise fall back to the first value
                if 0 <= idx - 1 < len(keys):
                    return raw_params[keys[idx - 1]]
                return raw_params.get("client_company_id", next(iter(raw_params.values())))

            ordered_markers = [int(m) for m in re.findall(r"\$(\d+)", raw_sql)]
            values = tuple(value_for_index(i) for i in ordered_markers)
            return re.sub(r"\$\d+", "%s", raw_sql), values
        return raw_sql, raw_params

    sql, exec_params = normalize_sql_params(sql, params)
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, exec_params or {})
            rows = cur.fetchall()
    # Basic normalization for JSON serialization
    def norm(v):
        if isinstance(v, (Decimal, _np.number)):
            return float(v)
        if isinstance(v, (_dt.date, _dt.datetime)):
            return v.isoformat()
        return v

    return [dict((k, norm(v)) for k, v in r.items()) for r in rows]


def macro_llm_agent(user_query: str, sector_name: str | None) -> Dict[str, Any]:
    """
    Use Tavily + Anthropic to get compact macro context for the sector/company.
    """
    global TRACE_LAST
    query = STATIC_MACRO_QUERY if MACRO_MODE == "static" else f"{sector_name or ''} Saudi Arabia macro {user_query}"
    tav = tavily_search(query, max_results=5)
    snippets = []
    for r in tav.get("results", []):
        title = r.get("title", "")
        snip = (r.get("content", "") or "")[:400]
        snippets.append(f"- {title}: {snip}")
    context = "\n".join(snippets) or "No macro results."

    aclient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    prompt = f"""
You are a macro analyst.

We have this macro search context:

{context}

Given the user question:

{user_query!r}

Summarize 3-5 key macro factors from this context that are most relevant to this company's performance
and assign each a rough polarity: positive, neutral, or negative.

Return JSON with:
- "summary": string
- "drivers": [{{"name": str, "polarity": "positive"|"negative"|"neutral", "comment": str}}]
"""
    t0 = time.perf_counter()
    resp = aclient.messages.create(
        model=os.getenv("CLAUDE_MODEL"),
        max_tokens=600,
        temperature=0.1,
        system="You only output valid JSON for a macro context summary.",
        messages=[{"role": "user", "content": prompt}],
    )
    dt = int((time.perf_counter() - t0) * 1000)
    raw = "".join(
        [b.text for b in resp.content if getattr(b, "type", None) == "text"]  # type: ignore[attr-defined]
    ).strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))

    TRACE_LAST["macro"] = {"provider": "anthropic", "model": resp.model, "latency_ms": dt, "raw": raw[:1000]}
    return data


import json, re, time, httpx, requests
from anthropic import Anthropic


def health_llm_agent(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Have an LLM sanity-check the shape of the numeric data / time series.
    """
    global TRACE_LAST
    aclient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    preview = json.dumps(rows[:24], indent=2)
    prompt = f"""
You are a senior data quality engineer.

We have a time series of quarterly company financial metrics (sample below, first up to 24 rows):

{preview}

Check for obvious issues:
- Missing quarters or large gaps.
- Zero or negative revenue where it looks implausible.
- Weird jumps in revenue or margins.

Return JSON with:
- "status": "ok" | "warning" | "error"
- "issues": [string]
- "notes": string
"""
    t0 = time.perf_counter()
    resp = aclient.messages.create(
        model=os.getenv("CLAUDE_MODEL"),
        max_tokens=600,
        temperature=0,
        system="You only output valid JSON for a data quality summary.",
        messages=[{"role": "user", "content": prompt}],
    )
    dt = int((time.perf_counter() - t0) * 1000)
    raw = "".join(
        [b.text for b in resp.content if getattr(b, "type", None) == "text"]  # type: ignore[attr-defined]
    ).strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    TRACE_LAST["health"] = {"provider": "anthropic", "model": resp.model, "latency_ms": dt, "raw": raw[:1200]}
    return data


from anthropic import Anthropic


def strategy_llm_agent(
    user_query: str,
    rows: List[Dict[str, Any]],
    health: Dict[str, Any],
    macro: Dict[str, Any],
    sector_name: str | None,
) -> Dict[str, Any]:
    """
    Turn metrics + macro into a prioritized action plan.
    """
    global TRACE_LAST
    aclient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    rows_preview = json.dumps(rows[-16:], indent=2)
    macro_summary = macro.get("summary", "")
    macro_drivers = macro.get("drivers", [])
    health_status = health.get("status", "unknown")
    health_issues = health.get("issues", [])

    prompt = f"""
You are a senior strategy consultant for a Saudi {sector_name or "company"}.

User question:
{user_query!r}

We have:
- Recent quarterly financials (last rows):
{rows_preview}

- Data quality assessment:
  - status: {health_status}
  - issues: {health_issues}

- Macro context:
  - summary: {macro_summary}
  - drivers: {macro_drivers}

Task:
1. Provide a one-sentence overall stance (e.g. "Cautiously positive", "Under pressure with margin risk").
2. Provide 3-7 concrete recommended actions for the next 90 days, specific and operational.
3. Provide 3-5 key signals to monitor (with rationale).

Return JSON with:
- "stance": string
- "actions": [string]
- "signals": {{"financial": [string], "operational": [string], "macro": [string]}}
"""
    t0 = time.perf_counter()
    resp = aclient.messages.create(
        model=os.getenv("STRATEGY_MODEL"),
        max_tokens=900,
        temperature=0.3,
        system="You only output valid JSON for a strategic recommendation summary.",
        messages=[{"role": "user", "content": prompt}],
    )
    dt = int((time.perf_counter() - t0) * 1000)
    raw = "".join(
        [b.text for b in resp.content if getattr(b, "type", None) == "text"]  # type: ignore[attr-defined]
    ).strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    TRACE_LAST["strategy_llm_primary"] = {
        "provider": "anthropic",
        "model": resp.model,
        "latency_ms": dt,
        "raw": raw[:1000],
    }
    return data


# === LangGraph wiring ===
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, List, Dict


class State(TypedDict, total=False):
    user_query: str
    client_company_id: int | str
    sql: str
    params: Dict[str, Any]
    rows: List[Dict[str, Any]]
    health_rows: List[Dict[str, Any]]
    health: Dict[str, Any]
    macro: Dict[str, Any]
    sector_name: str | None
    strategy: Dict[str, Any]
    final_answer: str
    short_answer: str
    trace: Dict[str, Any]
    error: str


def node_build_sql(state: State) -> State:
    if not USE_SQL_LLM:
        state["sql"] = """
        SELECT
          ci.client_company_id,
          ci.year::INT AS year,
          regexp_replace(concat('', ci.quarter), '[^0-9]', '', 'g')::INT AS quarter,
          (make_date(ci.year::INT, (regexp_replace(concat('', ci.quarter), '[^0-9]', '', 'g')::INT)*3, 1)
           + INTERVAL '2 months')::DATE AS period_end_date,
          ci.total_revenue,
          ci.operating_income,
          ci.ebitda,
          ci.interest_expense
        FROM clientincomestatements ci
        WHERE ci.client_company_id = %(client_company_id)s
        ORDER BY year, quarter;
        """.strip()
        state["params"] = {"client_company_id": state["client_company_id"]}
        return state

    out = sql_llm_agent(state["user_query"], state["client_company_id"])
    state["sql"] = out.get("sql", "")
    state["params"] = out.get("params", {"client_company_id": state["client_company_id"]})
    return state


def node_run_sql(state: State) -> State:
    rows = run_safe_query(state["sql"], state.get("params") or {})
    state["rows"] = rows
    return state


def node_health(state: State) -> State:
    rows = state.get("rows", [])
    if not rows:
        state["health"] = {"status": "warning", "issues": ["No rows returned"], "notes": ""}
        return state
    h = health_llm_agent(rows)
    state["health"] = h
    return state


def node_macro(state: State) -> State:
    m = macro_llm_agent(state.get("user_query", ""), state.get("sector_name"))
    state["macro"] = m
    return state


def build_narratives(rows: List[Dict[str, Any]], macro: Dict[str, Any]) -> tuple[str, str]:
    """
    Lightweight deterministic narrative for "final_answer" and a shorter summary.
    """
    if not rows:
        return ("No financial rows were returned for this client_company_id.", "No data.")
    # Very simple: last period values + macro summary
    last = rows[-1]
    rev = last.get("total_revenue") or last.get("revenue")
    op_inc = last.get("operating_income") or last.get("op_inc")
    year = last.get("year")
    qtr = last.get("quarter")
    macro_sum = macro.get("summary", "")
    full = (
        f"In Q{qtr} {year}, revenue was {rev:,.1f} and operating income was {op_inc:,.1f}. "
        f"Macro backdrop: {macro_sum or 'no specific macro commentary was retrieved.'}"
    )
    short = f"Q{qtr} {year}: revenue {rev:,.1f}, operating income {op_inc:,.1f}."
    return full, short


def node_strategy(state: State) -> State:
    strat = strategy_llm_agent(
        state.get("user_query", ""),
        state.get("rows", []),
        state.get("health", {}),
        state.get("macro", {}),
        state.get("sector_name"),
    )
    state["strategy"] = {
        "stance": strat.get("stance"),
        "actions": (strat.get("actions") or [])[:5],
        "signals": strat.get("signals", {}),
    }
    return state


def node_narrative(state: State) -> State:
    narrative_en, narrative_short = build_narratives(state.get("rows", []), state.get("macro", {}))
    state["final_answer"] = narrative_en
    state["short_answer"] = narrative_short
    return state


def node_finish(state: State) -> State:
    state["trace"] = get_trace()
    return state


# Build graph
builder = StateGraph(State)
builder.add_node("build_sql", node_build_sql)
builder.add_node("run_sql", node_run_sql)
builder.add_node("health", node_health)
builder.add_node("macro", node_macro)
builder.add_node("strategy", node_strategy)
builder.add_node("narrative", node_narrative)
builder.add_node("finish", node_finish)

builder.set_entry_point("build_sql")
builder.add_edge("build_sql", "run_sql")
builder.add_edge("run_sql", "health")
builder.add_edge("run_sql", "macro")
builder.add_edge("health", "strategy")
builder.add_edge("macro", "strategy")
builder.add_edge("strategy", "narrative")
builder.add_edge("narrative", "finish")
builder.set_finish_point("finish")

app = builder.compile()


def ask(query: str, *, client_company_id: int | str):
    reset_trace()
    init: State = {"user_query": query, "client_company_id": client_company_id}
    final: State = app.invoke(init)
    return final


def render_provenance(out: dict):
    tr = out.get("trace", {})
    print("=== Provenance ===")
    if tr.get("sql_agent"):
        sa = tr["sql_agent"]
        print(
            f"SQL Agent -> provider={sa.get('provider')} model={sa.get('model')} "
            f"latency={sa.get('latency_ms')}ms fallback={tr.get('fallback_used')}"
        )
        if sa.get("rationale"):
            print("Rationale:", str(sa["rationale"])[:400])
        if sa.get("json_fragment"):
            print("JSON fragment:", sa["json_fragment"][:400], "...")
    else:
        print(f"SQL Agent: (none)  fallback={tr.get('fallback_used')}")
    if tr.get("health"):
        h = tr["health"]
        print(
            f"Health LLM -> provider={h.get('provider')} model={h.get('model')} "
            f"latency={h.get('latency_ms')}ms"
        )
    if tr.get("macro"):
        m = tr["macro"]
        print(
            f"Macro LLM -> provider={m.get('provider')} model={m.get('model')} "
            f"latency={m.get('latency_ms')}ms"
        )
    if tr.get("strategy_llm_primary"):
        s = tr["strategy_llm_primary"]
        print(
            f"Strategy LLM -> provider={s.get('provider')} model={s.get('model')} "
            f"latency={s.get('latency_ms')}ms"
        )

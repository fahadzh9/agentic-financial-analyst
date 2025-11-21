import os
import re
import json
import time
import datetime as _dt
from decimal import Decimal
from typing import Any, Dict, List, TypedDict

import anthropic
import httpx
import numpy as _np
import psycopg
import requests
from anthropic import Anthropic
from langgraph.graph import StateGraph, END
from psycopg.rows import dict_row

# === Environment & feature toggles ===
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["CLAUDE_MODEL"] = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
os.environ["STRATEGY_MODEL"] = os.getenv("STRATEGY_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"))

USE_SQL_LLM = True
ENABLE_TRACE = True
MACRO_MODE = os.getenv("MACRO_MODE", "llm")  # "llm" or "static"
STATIC_MACRO_QUERY = "Saudi Arabia macro consumer demand inflation retail confidence policy latest"

# Clear proxies to avoid httpx proxy errors in some deployments
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(_k, None)

# === Database config ===
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
        host=PG["host"], port=PG["port"], dbname=PG["dbname"], user=PG["user"], password=PG["password"]
    )


# === Introspection ===
from typing import Optional


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
for _r in meta:
    by_table_cols.setdefault(_r["table_name"], []).append(_r["column_name"])

MARKET_TABLES = ["saudimarketcompanies", "balancesheets", "incomestatements", "sectors"]
CLIENT_TABLES = ["clients", "clientcompanies", "clientbalancesheets", "clientincomestatements"]


def pick(cols: List[str], cands: List[str]):
    lc = {c.lower(): c for c in cols}
    for p in cands:
        if p.lower() in lc:
            return lc[p.lower()]
    for p in cands:
        for c in cols:
            if p.lower() in c.lower():
                return c
    return None


def infer_client_roles(by: Dict[str, List[str]]):
    t = "clientincomestatements" if "clientincomestatements" in by else next((x for x in by if "income" in x and "client" in x), "clientincomestatements")
    cols = by.get(t, [])
    comp = by.get("clientcompanies", [])
    return {
        "table": t,
        "client_company_id": pick(cols, ["client_company_id", "clientcompanyid", "company_id", "client_id"]) or "client_company_id",
        "company_name": pick(comp, ["company_name", "name"]) or "company_name",
        "year": pick(cols, ["year"]) or "year",
        "quarter": pick(cols, ["quarter", "qtr"]) or "quarter",
        "revenue": pick(cols, ["total_revenue", "revenue", "net_sales", "sales"]) or "total_revenue",
        "op_inc": pick(cols, ["operating_income", "operating_profit", "total_operating_income_as_reported"]) or "operating_income",
    }


def infer_market_roles(by: Dict[str, List[str]]):
    t = "incomestatements" if "incomestatements" in by else next((x for x in by if "income" in x), "incomestatements")
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
        "op_inc": pick(cols, ["operating_income", "operating_profit", "total_operating_income_as_reported"]) or "operating_income",
    }


ROLES = {"client": infer_client_roles(by_table_cols), "market": infer_market_roles(by_table_cols)}


def build_schema_hint():
    lines = [
        "Result contract (REQUIRED aliases in final SELECT):",
        "- year AS year (INT)",
        "- quarter AS quarter (INT; if 'Q1'..'Q4' normalize with regexp_replace)",
        "- total_revenue AS revenue (NUMERIC)",
        "- operating_income AS operating_income (NUMERIC)",
        "- period_end_date AS period_end_date (DATE)",
        "- series AS series (TEXT: 'client' or 'market')",
        "- entity_id AS entity_id (TEXT), entity_label AS entity_label (TEXT)",
        "",
        "Tables:",
    ]
    order = [t for t in MARKET_TABLES if t in by_table_cols] + [t for t in CLIENT_TABLES if t in by_table_cols] + [t for t in by_table_cols if t not in MARKET_TABLES + CLIENT_TABLES]
    for t in order:
        lines.append(f"- {t}(" + ", ".join(by_table_cols[t]) + ")")
    rel = [
        "Relationships:",
        "- Market: incomestatements.(company_id, year, quarter, total_revenue, operating_income) ↔ saudimarketcompanies.(company_id, ticker, company_name, sector_id)",
        "- Client: clientincomestatements.(client_company_id, year, quarter, total_revenue, operating_income) ↔ clientcompanies.(client_company_id, company_name, sector_id)",
        "- Benchmark mapping: join clientcompanies.company_name = saudimarketcompanies.company_name (lowercased).",
    ]
    rules = [
        "Policy:",
        "- SELECT-only. NO DDL/DML.",
        "- MUST include %(client_company_id)s (never client name).",
        "- ORDER BY year DESC, quarter DESC; LIMIT ≤ 200.",
        "- Keep SQL simple, auditable.",
    ]
    return "\n".join(lines + rel + rules)


SCHEMA_HINT = build_schema_hint()

# === SQL guardrails ===
SQL_DEFAULT_LIMIT = 200


def inject_limit(sql: str, default_limit: int = SQL_DEFAULT_LIMIT) -> str:
    up = sql.upper()
    if " LIMIT " in up or " OFFSET " in up or "FETCH NEXT " in up:
        return sql
    return sql.rstrip().rstrip(";") + f" LIMIT {default_limit};"


def safe_execute(sql: str, params: Dict[str, Any] | None = None):
    up = sql.strip().upper()
    if any(up.startswith(x) for x in ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "TRUNCATE ", "CREATE "]):
        raise ValueError("Unsafe SQL blocked by guardrails")
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params or {})
            return cur.fetchall()


# === SQL Agent with fallback ===
SYSTEM_MSG = (
    "You are a senior financial SQL generator.\n"
    "Return STRICT JSON with keys: sql (string), params (object), rationale (string).\n"
    "Result contract: Final SELECT MUST alias the following columns exactly:\n"
    "[year, quarter, revenue, operating_income, period_end_date, series, entity_id, entity_label].\n\n"
    "CRITICAL RULES for 'quarter':\n"
    "• Many tables store quarter like 'Q1','Q2', or text. NEVER do arithmetic on raw quarter.\n"
    "• ALWAYS compute QUARTER_INT as: regexp_replace(concat('', <alias>.<quarter>), '[^0-9]', '', 'g')::INT\n"
    "  and use QUARTER_INT for any math or MAKE_DATE.\n"
    "• Example for period_end_date: make_date(year::INT, QUARTER_INT*3, 1) + INTERVAL '2 months'\n\n"
    "General Rules:\n"
    "- SELECT-only. LIMIT <= 200.\n"
    "- MUST include %(client_company_id)s parameter (never client name).\n"
    "- Benchmarking allowed by joining clientcompanies.company_name to saudimarketcompanies.company_name (lowercased).\n"
    "- ORDER BY year DESC, quarter DESC. Keep SQL simple, auditable.\n"
)

TRACE_LAST: Dict[str, Any] = {"sql_agent": {}, "health": {}, "macro_prompt": {}, "strategy_llm_primary": {}, "fallback_used": False}


def _trace_reset():
    global TRACE_LAST
    TRACE_LAST = {"sql_agent": {}, "health": {}, "macro_prompt": {}, "strategy_llm_primary": {}, "fallback_used": False}


def call_claude_json(query: str) -> str:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
        max_tokens=1200,
        temperature=0,
        system=SYSTEM_MSG,
        messages=[{"role": "user", "content": json.dumps({"question": query, "schema_hint": SCHEMA_HINT}, ensure_ascii=False)}],
    )
    latency = int((time.perf_counter() - t0) * 1000)
    text = "".join([b.text for b in resp.content if b.type == "text"])
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Claude returned no JSON object")
    js = m.group(0)
    if ENABLE_TRACE:
        TRACE_LAST["sql_agent"] = {
            "provider": "anthropic",
            "model": os.getenv("CLAUDE_MODEL"),
            "latency_ms": latency,
            "raw": text[:2000],
            "json_fragment": js[:1000],
        }
    return js


def _maybe_order_by_yq(sql: str) -> str:
    low = sql.lower()
    if (" order by " not in low) and ((" year " in low) or ("year::" in low)) and ((" quarter " in low) or ("quarter::" in low)):
        return sql.rstrip().rstrip(";") + " ORDER BY year DESC, quarter DESC"
    return sql


def fallback_sql() -> str:
    rc = ROLES["client"]
    rm = ROLES["market"]
    return (
        f"WITH mine AS ("
        f" SELECT ci.{rc['year']}::INT AS year,"
        f"        regexp_replace(concat('', ci.{rc['quarter']}), '[^0-9]', '', 'g')::INT AS quarter,"
        f"        ci.{rc['revenue']}::NUMERIC AS revenue, ci.{rc['op_inc']}::NUMERIC AS operating_income,"
        f"        (make_date(ci.{rc['year']}::INT, (regexp_replace(concat('', ci.{rc['quarter']}), '[^0-9]', '', 'g')::INT)*3, 1) + INTERVAL '2 months')::DATE AS period_end_date,"
        f"        'client'::TEXT AS series, ci.{rc['client_company_id']}::TEXT AS entity_id, cc.{rc['company_name']}::TEXT AS entity_label"
        f" FROM {rc['table']} ci JOIN clientcompanies cc ON cc.{rc['client_company_id']}=ci.{rc['client_company_id']}"
        f" WHERE ci.{rc['client_company_id']}=%(client_company_id)s"
        f" ORDER BY ci.{rc['year']} DESC, regexp_replace(concat('', ci.{rc['quarter']}), '[^0-9]', '', 'g')::INT DESC"
        f" LIMIT 8),"
        f" map_market AS ("
        f" SELECT sm.company_id AS market_company_id FROM clientcompanies cc"
        f" JOIN saudimarketcompanies sm ON LOWER(cc.{rc['company_name']}) = LOWER(sm.{rm['company_name']})"
        f" WHERE cc.{rc['client_company_id']}=%(client_company_id)s LIMIT 1),"
        f" market AS ("
        f" SELECT mi.{rm['year']}::INT AS year,"
        f"        regexp_replace(concat('', mi.{rm['quarter']}), '[^0-9]', '', 'g')::INT AS quarter,"
        f"        mi.{rm['revenue']}::NUMERIC AS revenue, mi.{rm['op_inc']}::NUMERIC AS operating_income,"
        f"        (make_date(mi.{rm['year']}::INT, (regexp_replace(concat('', mi.{rm['quarter']}), '[^0-9]', '', 'g')::INT)*3, 1) + INTERVAL '2 months')::DATE AS period_end_date,"
        f"        'market'::TEXT AS series, sm.{rm['ticker']}::TEXT AS entity_id, sm.{rm['company_name']}::TEXT AS entity_label"
        f" FROM {rm['table']} mi JOIN saudimarketcompanies sm ON sm.company_id=mi.{rm['company_id']}"
        f" WHERE mi.{rm['company_id']}=(SELECT market_company_id FROM map_market)"
        f" ORDER BY mi.{rm['year']} DESC, regexp_replace(concat('', mi.{rm['quarter']}), '[^0-9]', '', 'g')::INT DESC"
        f" LIMIT 8)"
        f" SELECT * FROM mine UNION ALL SELECT * FROM market ORDER BY year DESC, quarter DESC;"
    )


def generate_sql(query: str) -> str:
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("CLAUDE_MODEL"):
        TRACE_LAST["fallback_used"] = True
        return fallback_sql()
    try:
        obj = json.loads(call_claude_json(query))
        sql = (obj.get("sql") or "").strip()
        TRACE_LAST["sql_agent"]["rationale"] = obj.get("rationale")

        up = sql.upper()
        if (not sql) or any(up.startswith(x) for x in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]):
            TRACE_LAST["fallback_used"] = True
            return fallback_sql()
        if "%(client_company_id)s" not in sql or "%(client_name)s" in sql:
            TRACE_LAST["fallback_used"] = True
            return fallback_sql()

        sql = _maybe_order_by_yq(sql)
        if " LIMIT " not in up:
            sql = inject_limit(sql)
        return sql
    except Exception as e:  # noqa: BLE001
        TRACE_LAST["fallback_used"] = True
        TRACE_LAST["sql_agent"]["error"] = str(e)
        return fallback_sql()


# === JSON helpers ===

def make_json_safe(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "ignore")
    if isinstance(obj, (_np.integer, _np.floating)):
        return float(obj)
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    return obj


def canonicalize_rows(rows: List[Dict[str, Any]]):
    out = []
    for r in rows:
        c = dict(r)
        if "revenue" not in c and "total_revenue" in c:
            c["revenue"] = c.get("total_revenue")
        if "operating_income" not in c:
            for k in ["op_income", "total_operating_income_as_reported", "operating_profit"]:
                if k in c:
                    c["operating_income"] = c.get(k)
        if "quarter" not in c:
            for k in ["quarter_num", "qtr", "qtr_num"]:
                if k in c:
                    c["quarter"] = c.get(k)
        if "series" not in c:
            c["series"] = "client"
        out.append(c)
    return out


# === Sector helper ===

def get_sector_name(client_company_id: int | str) -> Optional[str]:
    q = """
    SELECT s.sector_name
    FROM clientcompanies c
    LEFT JOIN sectors s ON s.sector_id = c.sector_id
    WHERE c.client_company_id = %(client_company_id)s
    LIMIT 1;
    """
    rows = safe_execute(q, {"client_company_id": client_company_id})
    return (rows[0].get("sector_name") if rows else None) or None


OIL_RELEVANT_SECTORS = {
    "OIL_GAS",
    "ENERGY",
    "PETROCHEMICALS",
    "CHEMICALS",
    "AIRLINES",
    "SHIPPING",
    "LOGISTICS",
    "TRANSPORT",
}


# === Macro prompt agent + Tavily ===

def _sanitize_macro_query(s: str, max_len: int = 180) -> str:
    s = re.sub(r"[^A-Za-z0-9\s\-,/():&]+", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def _sector_default_query(sector: Optional[str]) -> str:
    s = (sector or "").upper()
    if s in OIL_RELEVANT_SECTORS:
        return "Saudi Arabia energy demand fuel prices refinery margins shipping freight latest"
    if "FOOD" in s or "BEVERAGE" in s:
        return "Saudi Arabia food & beverage demand inflation FAO Food Price Index wheat sugar palm oil packaging freight latest"
    if "RETAIL" in s or "CONSUMER" in s:
        return "Saudi Arabia consumer spending retail sales inflation employment confidence latest"
    if "CEMENT" in s or "BUILD" in s or "CONSTRUCTION" in s:
        return "Saudi Arabia construction cement demand input costs freight policy projects latest"
    return "Saudi Arabia macro consumer demand inflation retail confidence policy latest"


def macro_prompt_agent(user_query: str, rows: List[Dict[str, Any]], health: Dict[str, Any], sector_name: Optional[str]) -> Dict[str, Any]:
    brief = {}
    try:
        last = (rows or [None])[0] or {}
        brief = {
            "sector": sector_name,
            "latest_quarter": str(last.get("quarter")),
            "latest_year": str(last.get("year")),
            "latest_rev": last.get("total_revenue") or last.get("revenue"),
            "latest_op": last.get("operating_income"),
            "health_verdict": (health or {}).get("verdict"),
            "health_score": (health or {}).get("score"),
        }
    except Exception:  # noqa: BLE001
        brief = {}

    system = (
        "You write ONE concise web search query for macro context that is RELEVANT TO THE SECTOR.\n"
        "Rules:\n"
        "- Prefer the company's sector; avoid generic oil unless sector is directly oil-exposed (energy, petrochem, airlines, shipping, transport).\n"
        "- For FOOD/BEVERAGE emphasize: FAO Food Price Index, wheat/sugar/palm oil, packaging/plastics, freight, consumer demand/retail.\n"
        "- 6–16 keywords; include 'Saudi Arabia' and 'latest' when helpful.\n"
        "Output STRICT JSON: {\"query\":\"...\", \"tags\":[oil_prices,rates_fx,inflation,demand,supply_costs,policy,food_inputs,freight]}\n"
        "If unsure, avoid oil and focus on demand/inflation/sector inputs."
    )

    aclient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    payload = {"user_question": user_query, "signals": make_json_safe(brief)}
    t0 = time.perf_counter()
    resp = aclient.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
        max_tokens=200,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    latency = int((time.perf_counter() - t0) * 1000)
    raw = "".join([b.text for b in resp.content if b.type == "text"])
    try:
        j = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        j = {"query": _sector_default_query(sector_name), "tags": []}

    q = _sanitize_macro_query(j.get("query") or _sector_default_query(sector_name))
    tags = [t for t in (j.get("tags") or []) if isinstance(t, str)]

    s_up = (sector_name or "").upper()
    if s_up and s_up not in OIL_RELEVANT_SECTORS:
        tags = [t for t in tags if t != "oil_prices"]
        if ("FOOD" in s_up or "BEVERAGE" in s_up) and "food_inputs" not in tags:
            tags.append("food_inputs")

    if ENABLE_TRACE:
        TRACE_LAST["macro_prompt"] = {
            "provider": "anthropic",
            "model": os.getenv("CLAUDE_MODEL"),
            "latency_ms": latency,
            "query": q,
            "tags": tags,
            "raw": raw[:400],
        }
    return {"query": q, "tags": tags}


def fetch_macro_dynamic(user_query: str, client_company_id: str | int, rows: List[Dict[str, Any]], health: Dict[str, Any], sector_name: Optional[str], max_results: int = 5):
    key = os.getenv("TAVILY_API_KEY", "")
    if MACRO_MODE == "llm":
        mp = macro_prompt_agent(user_query, rows, health, sector_name)
        q = mp.get("query") or _sector_default_query(sector_name)
        llm_tags = mp.get("tags") or []
    else:
        q = _sector_default_query(sector_name)
        llm_tags = []

    if not key:
        return {"warning": "TAVILY_API_KEY not set; macro skipped.", "query": q, "llm_tags": llm_tags, "results": []}
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": q, "max_results": max_results, "search_depth": "basic", "include_answer": False},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        results = [
            {"title": it.get("title"), "url": it.get("url"), "content": (it.get("content") or "")[:300]}
            for it in data.get("results", [])[:max_results]
        ]
        return {"query": q, "results": results, "llm_tags": llm_tags}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "query": q, "llm_tags": llm_tags, "results": []}


# === Narrative builder ===
REV_KEYS = ["revenue", "total_revenue"]
OPI_KEYS = ["operating_income", "op_income", "total_operating_income_as_reported"]


def _abbr_num(x):
    try:
        n = float(x)
    except Exception:  # noqa: BLE001
        return None
    a = abs(n)
    if a >= 1e9:
        return f"{n/1e9:.1f}B"
    if a >= 1e6:
        return f"{n/1e6:.1f}M"
    if a >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _first_key(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _q_to_int(q):
    if q is None:
        return None
    if isinstance(q, (int, float)):
        try:
            return int(q)
        except Exception:  # noqa: BLE001
            return None
    m = re.search(r"(\d+)", str(q))
    return int(m.group(1)) if m else None


def _as_float(x):
    try:
        return None if x is None else float(x)
    except Exception:  # noqa: BLE001
        return None


def _pct(a, b):
    try:
        if b in (None, 0) or a is None:
            return None
        return (float(a) - float(b)) / abs(float(b)) * 100.0
    except Exception:  # noqa: BLE001
        return None


def _margin(row: Dict[str, Any]):
    rev = _as_float(_first_key(row, REV_KEYS))
    op = _as_float(_first_key(row, OPI_KEYS))
    if rev in (None, 0) or op is None:
        return None
    return (op / rev) * 100.0


def _sort_key(row: Dict[str, Any]):
    ped = row.get("period_end_date")
    if ped is not None:
        return ped, 5
    y = _first_key(row, ["year", "fiscal_year"])
    q = _q_to_int(_first_key(row, ["quarter", "qtr", "quarter_num", "qtr_num"]))
    return y, q if q is not None else -1


def _peer_for_period(rows: List[Dict[str, Any]], y, q):
    for r in rows:
        if r.get("series") == "market" and r.get("year") == y and _q_to_int(r.get("quarter")) == q:
            return r
    return None


def build_narratives(rows: List[Dict[str, Any]], macro: Dict[str, Any]):
    if not rows:
        return "No recent data.", "No recent data."
    rows = canonicalize_rows(rows)
    client_rows = [r for r in rows if r.get("series") == "client"] or rows
    client_rows = sorted(client_rows, key=_sort_key, reverse=True)

    latest = client_rows[0]
    prev = client_rows[1] if len(client_rows) > 1 else None

    y = latest.get("year")
    q = _q_to_int(latest.get("quarter"))
    rev = _as_float(_first_key(latest, REV_KEYS))
    rev_prev = _as_float(_first_key(prev, REV_KEYS)) if prev else None
    op = _as_float(_first_key(latest, OPI_KEYS))

    growth = _pct(rev, rev_prev) if prev else None
    margin = _margin(latest)

    peer = _peer_for_period(rows, y, q)
    peer_mrg = _margin(peer) if peer else None
    spread = None if (margin is None or peer_mrg is None) else (margin - peer_mrg)

    macro_titles = [(it.get("title") or "").strip() for it in (macro.get("results") or [])[:2] if (it.get("title") or "").strip()]
    macro_part = f" On the macro side, we’re watching: {', '.join(macro_titles)}." if macro_titles else ""

    if rev is not None and margin is not None and growth is not None:
        dir_en = "up" if growth >= 0 else "down"
        rev_fmt = _abbr_num(rev) or f"{rev:,.0f}"
        peer_txt = f" (market {peer_mrg:.1f}%, spread {spread:+.1f} pp)" if (peer_mrg is not None and spread is not None) else ""
        en = f"In Q{q} {y}, revenue reached {rev_fmt}, {dir_en} {abs(growth):.1f}% vs last quarter. Operating margin was {margin:.1f}%{peer_txt}.{macro_part}"
    else:
        parts = []
        if rev is not None:
            rev_fmt = _abbr_num(rev) or f"{rev:,.0f}"
            base = f"In Q{q} {y}, revenue was {rev_fmt}"
            if growth is not None:
                dir_en = "up" if growth >= 0 else "down"
                base += f", {dir_en} {abs(growth):.1f}% vs last quarter"
            parts.append(base + ".")
        if margin is not None:
            peer_txt = f" (market {peer_mrg:.1f}%, spread {spread:+.1f} pp)" if (peer_mrg is not None and spread is not None) else ""
            parts.append(f"Operating margin stood at {margin:.1f}%{peer_txt}.")
        if not parts:
            parts = ["We couldn’t compute a clean comparison for the latest quarter."]
        en = " ".join(parts) + macro_part

    short_bits = [
        f"Revenue {('+' if (growth is not None and growth >= 0) else '')}{growth:.1f}% QoQ" if growth is not None else "Revenue n/a QoQ",
        f"Margin {margin:.1f}%" if margin is not None else "Margin n/a",
    ]
    if macro_titles:
        short_bits.append("Macro: " + ", ".join(macro_titles))
    short = " • ".join(short_bits)
    return en, short


# === Health SQL + agent ===
HEALTH_SQL = """
WITH inc AS (
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
),
bs AS (
  SELECT
    cb.client_company_id,
    cb.year::INT AS year,
    regexp_replace(concat('', cb.quarter), '[^0-9]', '', 'g')::INT AS quarter,
    (make_date(cb.year::INT, (regexp_replace(concat('', cb.quarter), '[^0-9]', '', 'g')::INT)*3, 1)
     + INTERVAL '2 months')::DATE AS period_end_date,
    cb.total_assets,
    cb.current_assets,
    cb.current_liabilities,
    cb.cash_and_cash_equivalents,
    cb.total_debt,
    cb.stockholders_equity,
    cb.net_debt
  FROM clientbalancesheets cb
  WHERE cb.client_company_id = %(client_company_id)s
),
merged AS (
  SELECT
    COALESCE(inc.client_company_id, bs.client_company_id) AS client_company_id,
    COALESCE(inc.year, bs.year) AS year,
    COALESCE(inc.quarter, bs.quarter) AS quarter,
    COALESCE(inc.period_end_date, bs.period_end_date) AS period_end_date,
    inc.total_revenue,
    inc.operating_income,
    inc.ebitda,
    inc.interest_expense,
    bs.total_assets,
    bs.current_assets,
    bs.current_liabilities,
    bs.cash_and_cash_equivalents,
    bs.total_debt,
    bs.stockholders_equity,
    bs.net_debt
  FROM inc
  FULL OUTER JOIN bs
    ON inc.client_company_id = bs.client_company_id
   AND inc.year = bs.year
   AND inc.quarter = bs.quarter
)
SELECT *
FROM merged
ORDER BY year DESC, quarter DESC
LIMIT 12;
"""


def get_health_rows(client_company_id: int | str):
    return safe_execute(HEALTH_SQL, params={"client_company_id": client_company_id})


def health_agent_llm(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ctx_rows = (rows or [])[:8]
    system = (
        "You are a senior credit analyst. "
        "Given up to 8 quarterly rows (latest first) with fields like year, quarter, period_end_date, "
        "total_revenue, operating_income, ebitda, interest_expense, total_assets, current_assets, "
        "current_liabilities, cash_and_cash_equivalents, total_debt, stockholders_equity, net_debt — "
        "compute a financial health summary.\n\n"
        "Output STRICT JSON with keys:\n"
        "{\n"
        '  "metrics": {\n'
        '    "d_to_e": number|null,\n'
        '    "interest_coverage": number|null,\n'
        '    "net_debt": number|null,\n'
        '    "net_debt_to_ebitda": number|null,\n'
        '    "current_ratio": number|null,\n'
        '    "quick_ratio": number|null,\n'
        '    "margin_pct": number|null,\n'
        '    "revenue_qoq_pct": number|null,\n'
        '    "revenue_yoy_pct": number|null\n'
        '  },\n'
        '  "verdict": "strong"|"moderate"|"elevated"|"weak",\n'
        '  "score": number,\n'
        '  "green_flags": [string],\n'
        '  "red_flags": [string],\n'
        '  "rationale": string\n'
        "}\n"
        "Rules: Use only provided numbers; if a metric cannot be computed, set null. Return JSON only."
    )
    aclient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    payload = {"rows": make_json_safe(ctx_rows)}
    t0 = time.perf_counter()
    resp = aclient.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
        max_tokens=900,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    latency = int((time.perf_counter() - t0) * 1000)
    raw = "".join([b.text for b in resp.content if getattr(b, "type", None) == "text"])
    j = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    if ENABLE_TRACE:
        TRACE_LAST["health"] = {"provider": "anthropic", "model": resp.model, "latency_ms": latency, "raw": raw[:1200]}
    return j


# === Strategy agent ===
FORBIDDEN_PHRASES = ["buy the stock", "short the stock", "invest now", "target price", "EPS upgrade", "equity raise", "issue shares", "IPO"]


def strategy_llm_agent(user_query: str, rows: List[Dict[str, Any]], health: Dict[str, Any], macro: Dict[str, Any], sector_name: Optional[str] = None) -> Dict[str, Any]:
    rows_ctx = (canonicalize_rows(rows) or [])[:8]
    macro_ctx = (macro.get("results") or [])[:3]
    macro_tags = macro.get("llm_tags", [])

    system = (
        "You are an operating strategy advisor.\n"
        f"Company sector: {sector_name or 'UNKNOWN'}.\n"
        "Tailor stance/actions to the user's question, using ONLY provided data (rows, health, macro tags/snippets).\n"
        "Constraints:\n"
        "- No investment advice (no trading, targets, etc.).\n"
        "- If sector is NOT oil-exposed, avoid oil commentary unless macro_tags includes 'oil_prices'.\n"
        "- Keep actions concrete and operational; ≤5 bullets.\n"
        "Output STRICT JSON ONLY:\n"
        "{\n"
        '  "stance": "constructive – ..." | "defensive – ..." | "neutral – ..." | "mixed – ...",\n'
        '  "actions": [string, string, string, string, string],\n'
        '  "signals": { "margin_pct": number|null, "rev_growth_qoq_pct": number|null, "health_score": number|null, "macro_tags": [string] }\n'
        "}"
    )

    aclient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), http_client=httpx.Client(timeout=60.0))
    payload = {
        "user_question": str(user_query or "").strip(),
        "sector": sector_name,
        "rows": make_json_safe(rows_ctx),
        "health": health,
        "macro": make_json_safe(macro_ctx),
        "macro_tags": macro_tags,
    }
    t0 = time.perf_counter()
    resp = aclient.messages.create(
        model=os.getenv("STRATEGY_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")),
        max_tokens=650,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    raw = "".join([b.text for b in resp.content if b.type == "text"])
    j = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])

    s_up = (sector_name or "").upper()
    if s_up and s_up not in OIL_RELEVANT_SECTORS and "oil_prices" not in (macro_tags or []):
        j["actions"] = [a for a in (j.get("actions") or []) if not re.search(r"\boil\b|\bbrent\b|\bopec\b", a, re.I)]

    acts = [a for a in (j.get("actions") or []) if isinstance(a, str)][:5]
    j["actions"] = acts
    if "signals" not in j:
        j["signals"] = {}
    if "macro_tags" not in j["signals"]:
        j["signals"]["macro_tags"] = macro_tags

    if ENABLE_TRACE:
        TRACE_LAST["strategy_llm_primary"] = {"provider": "anthropic", "model": resp.model, "latency_ms": latency_ms, "raw": raw[:1000]}
        TRACE_LAST["strategy_user_prompt"] = str(user_query)[:240]
    return j


# === LangGraph wiring ===
class State(TypedDict, total=False):
    user_query: str
    client_company_id: int | str
    sql: str
    params: Dict[str, Any]
    rows: List[Dict[str, Any]]
    health_rows: List[Dict[str, Any]]
    health: Dict[str, Any]
    macro: Dict[str, Any]
    sector_name: Optional[str]
    strategy: Dict[str, Any]
    final_answer: str
    short_answer: str
    trace: Dict[str, Any]
    error: str


def node_sql_agent(state: State) -> State:
    _trace_reset()
    q = state.get("user_query", "")
    ccid = state.get("client_company_id")
    assert ccid not in (None, ""), "client_company_id is required"

    sql = generate_sql(q) if USE_SQL_LLM else fallback_sql()
    if "%(client_company_id)s" not in sql or "%(client_name)s" in sql:
        raise AssertionError("SQL must use %(client_company_id)s only (no client_name).")

    try:
        rows = safe_execute(sql, {"client_company_id": ccid})
    except Exception as e:  # noqa: BLE001
        if ENABLE_TRACE:
            TRACE_LAST["sql_execute_error"] = str(e)
        sql_fb = fallback_sql()
        rows = safe_execute(sql_fb, {"client_company_id": ccid})
        sql = sql_fb
        TRACE_LAST["fallback_used"] = True

    return {"sql": sql, "params": {"client_company_id": ccid}, "rows": rows, "trace": TRACE_LAST}


def node_health(state: State) -> State:
    ccid = state["client_company_id"]
    health_rows = get_health_rows(ccid)
    try:
        health_json = health_agent_llm(health_rows)
    except Exception as e:  # noqa: BLE001
        health_json = {"error": str(e)}
        if ENABLE_TRACE:
            tr = state.get("trace", {})
            tr["health_error"] = str(e)
            state["trace"] = tr
    state["health_rows"] = health_rows
    state["health"] = health_json
    return state


def node_macro(state: State) -> State:
    mq = state.get("user_query", "")
    ccid = state["client_company_id"]
    sector_name = get_sector_name(ccid)
    macro = fetch_macro_dynamic(mq, ccid, state.get("rows", []), state.get("health", {}), sector_name)
    state["macro"] = macro
    state["sector_name"] = sector_name
    return state


def node_strategy(state: State) -> State:
    strat = strategy_llm_agent(
        state.get("user_query", ""),
        state.get("rows", []),
        state.get("health", {}),
        state.get("macro", {}),
        state.get("sector_name"),
    )
    state["strategy"] = {"stance": strat.get("stance"), "actions": (strat.get("actions") or [])[:5], "signals": strat.get("signals", {})}
    return state


def node_narrative(state: State) -> State:
    narrative_en, narrative_short = build_narratives(state.get("rows", []), state.get("macro", {}))
    state["final_answer"] = narrative_en
    state["short_answer"] = narrative_short
    return state


def node_finish(state: State) -> State:
    return {"trace": TRACE_LAST}


builder = StateGraph(State)
builder.add_node("sql_agent", node_sql_agent)
builder.add_node("health", node_health)
builder.add_node("macro", node_macro)
builder.add_node("strategy", node_strategy)
builder.add_node("narrative", node_narrative)
builder.add_node("finish", node_finish)

builder.set_entry_point("sql_agent")
builder.add_edge("sql_agent", "health")
builder.add_edge("health", "macro")
builder.add_edge("macro", "strategy")
builder.add_edge("strategy", "narrative")
builder.add_edge("narrative", "finish")
builder.set_finish_point("finish")

app = builder.compile()


def ask(query: str, *, client_company_id: int | str):
    _trace_reset()
    init: State = {"user_query": query, "client_company_id": client_company_id}
    final: State = app.invoke(init)
    return final


def render_provenance(out: dict):
    tr = out.get("trace", {})
    print("=== Provenance ===")
    if tr.get("sql_agent"):
        sa = tr["sql_agent"]
        print(f"SQL Agent -> provider={sa.get('provider')} model={sa.get('model')} latency={sa.get('latency_ms')}ms fallback={tr.get('fallback_used')}")
        if sa.get("rationale"):
            print("Rationale:", str(sa["rationale"])[:400])
        if sa.get("json_fragment"):
            print("JSON fragment:", sa["json_fragment"][:400], "...")
    else:
        print(f"SQL Agent: (none)  fallback={tr.get('fallback_used')}")
    if tr.get("health"):
        h = tr["health"]
        print(f"Health Agent -> provider={h.get('provider')} model={h.get('model')} latency={h.get('latency_ms')}ms")
    elif tr.get("health_error"):
        print("Health Agent -> error:", tr["health_error"])
    else:
        print("Health Agent: (none)")
    if tr.get("macro_prompt"):
        mp = tr["macro_prompt"]
        print(f"Macro Prompt Agent -> provider={mp.get('provider')} model={mp.get('model')} latency={mp.get('latency_ms')}ms")
        print("Macro query:", mp.get("query"), "| tags:", mp.get("tags"))
    else:
        print("Macro Prompt Agent: (none)")
    if tr.get("strategy_llm_primary"):
        st = tr["strategy_llm_primary"]
        print(f"Strategy LLM (primary) -> provider={st.get('provider')} model={st.get('model')} latency={st.get('latency_ms')}ms")
        if st.get("raw"):
            print("Raw (trunc):", st["raw"][:300])
    else:
        print("Strategy LLM (primary): (none)")
    if tr.get("strategy_user_prompt"):
        print("Strategy saw user prompt:", tr["strategy_user_prompt"])


def diagnostics():
    env_required = [
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "PGHOST",
        "PGPORT",
        "PGDATABASE",
        "PGUSER",
        "PGPASSWORD",
    ]
    env_status = {k: bool(os.getenv(k)) for k in env_required}

    db_status: dict[str, Any]
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                _ = cur.fetchone()
        db_status = {
            "status": "ok",
            "target": f"{PG['user']}@{PG['host']}:{PG['port']}/{PG['dbname']} ({PG['schema']})",
        }
    except Exception as e:  # noqa: BLE001
        db_status = {"status": "error", "message": str(e)}

    llm_status: dict[str, Any]
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    if not api_key:
        llm_status = {"status": "error", "message": "ANTHROPIC_API_KEY is missing"}
    else:
        try:
            # Use an explicit httpx client with trust_env=False to avoid
            # environments that inject proxy settings incompatible with the
            # Anthropic SDK (e.g., unexpected 'proxies' kwargs seen in Azure).
            with httpx.Client(timeout=10.0, trust_env=False) as http_client:
                client = Anthropic(api_key=api_key, http_client=http_client)
                resp = client.messages.create(
                    model=model,
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Reply with 'ok'"}],
                    system="You are a lightweight health check. Answer with 'ok'.",
                )
                reply = resp.content[0].text if resp.content else ""
                llm_status = {"status": "ok", "model": model, "reply": reply}
        except Exception as e:  # noqa: BLE001
            llm_status = {"status": "error", "model": model, "message": str(e)}

    return {
        "environment": env_status,
        "database": db_status,
        "llm": llm_status,
    }

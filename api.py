from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from financial_analyst_graph import ask, diagnostics  # your LangGraph app


app = FastAPI(
    title="Agentic Financial Analyst Service",
    version="1.0.0",
    description=(
        "Sector-aware agentic financial analyst built with LangGraph and Anthropic, "
        "exposed as an HTTP API."
    ),
)


# --------- Models ---------


class AskRequest(BaseModel):
    prompt: str
    client_company_id: int


class AskResponse(BaseModel):
    final_answer: str
    short_answer: Optional[str] = None
    strategy: Optional[Dict[str, Any]] = None
    rows: Optional[List[Dict[str, Any]]] = None
    trace: Optional[Dict[str, Any]] = None


# --------- Health / Diagnostics ---------


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/diagnostics")
def diagnostics_endpoint():
    """Check environment variables, database connectivity, and LLM health."""
    return diagnostics()


# --------- Main /ask endpoint ---------


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    """
    Invoke the financial analyst graph.

    Example request body:
    {
      "prompt": "Revenue and margin trend with market benchmark",
      "client_company_id": 6
    }
    """
    try:
        # This should already return a JSON-serializable dict:
        # {
        #   "final_answer": "...",
        #   "short_answer": "...",      # optional
        #   "strategy": {...},          # optional
        #   "rows": [...],              # optional
        #   "trace": {...}              # optional
        # }
        out = ask(
            payload.prompt,
            client_company_id=payload.client_company_id,
        )
        return out

    except Exception as e:
        # Convert any internal error into a clean JSON 500 for the client
        # (so you don't get HTML/502 surprises).
        detail: Dict[str, Any] = {
            "detail": f"Graph invocation failed: {type(e).__name__}: {e}",
        }

        # Optionally attach trace if your graph exposes it AND it's not too big
        try:
            from financial_analyst_graph import TRACE_LAST, make_json_safe  # type: ignore

            detail["trace"] = make_json_safe(TRACE_LAST)
        except Exception:
            # If TRACE_LAST or make_json_safe don't exist / fail, ignore silently
            pass

        raise HTTPException(status_code=500, detail=detail)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict

from financial_analyst_graph import ask, diagnostics  # uses the LangGraph app defined in the notebook

app = FastAPI(
    title="Agentic Financial Analyst Service",
    version="1.0.0",
    description="Sector-aware agentic financial analyst built with LangGraph and Anthropic, exposed as an HTTP API.",
)


class AskRequest(BaseModel):
    prompt: str
    client_company_id: int


class AskResponse(BaseModel):
    final_answer: str
    short_answer: str | None = None
    strategy: Dict[str, Any] | None = None
    rows: list[dict] | None = None
    trace: Dict[str, Any] | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/diagnostics")
async def diagnostics_endpoint():
    """Check environment variables, database connectivity, and LLM health."""
    return diagnostics()


@app.post("/ask")
async def ask_endpoint(payload: AskRequest):
    """
    Invoke the financial analyst graph.

    Request body:
    {
      "prompt": "Revenue and margin trend with market benchmark",
      "client_company_id": 6
    }
    """
    try:
        out = ask(payload.prompt, client_company_id=payload.client_company_id)
    except Exception as e:
        # Ensure the client always receives JSON, even when unexpected errors bubble up.
        # Include the latest trace snapshot when available to help with debugging.
        detail = {"detail": f"Graph invocation failed: {e}"}
        try:
            from financial_analyst_graph import TRACE_LAST, make_json_safe

            detail["trace"] = make_json_safe(TRACE_LAST)
        except Exception:
            pass

        return JSONResponse(status_code=500, content=detail)

    # FastAPI can JSON-encode most built-in types (including date/datetime) automatically,
    # but we already sanitized the graph output inside ask() for safety.
    return JSONResponse(content=out)

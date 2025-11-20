from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from financial_analyst_graph import ask  # uses the LangGraph app defined in the notebook

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
        # You might want to add proper structured logging here.
        raise HTTPException(status_code=500, detail=f"Graph invocation failed: {e}") from e

    # FastAPI can JSON-encode most built-in types (including date/datetime) automatically.
    return out

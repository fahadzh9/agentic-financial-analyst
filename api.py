from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

# Only import diagnostics for now so we don't change that behavior
from financial_analyst_graph import diagnostics

app = FastAPI(
    title="Agentic Financial Analyst Service",
    version="1.0.0",
    description="Sector-aware agentic financial analyst built with LangGraph and Anthropic, exposed as an HTTP API.",
)


class AskRequest(BaseModel):
    prompt: str
    client_company_id: int


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
    TEMPORARY: simple echo to test Azure / gunicorn / routing.
    """
    return {
        "status": "ok",
        "echo_prompt": payload.prompt,
        "echo_client_company_id": payload.client_company_id,
    }

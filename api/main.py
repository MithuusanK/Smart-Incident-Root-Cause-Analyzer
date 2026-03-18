"""
Smart Incident Root Cause Analyzer — FastAPI Application
POST /analyze          — analyze an incident
GET  /health           — health check
GET  /incidents        — list past analyses (paginated)
GET  /incidents/{id}   — get specific analysis
POST /feedback/{id}    — submit feedback on an analysis
GET  /stats            — aggregate statistics
POST /grafana/webhook  — Grafana alert webhook
"""

from dotenv import load_dotenv
load_dotenv()  # loads api/.env automatically

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import httpx
from database import (
    find_similar_incidents,
    get_analysis,
    get_db,
    get_stats,
    init_db,
    list_analyses,
    load_training_data,
    save_analysis,
    update_feedback,
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import get_analyzer
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Incident Root Cause Analyzer",
    description="AI-powered production incident analysis. Analyzes logs, metrics, and error traces to identify root causes in minutes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("Initialising database indexes...")
    await init_db()

    seed_path = os.environ.get("SEED_DATA_PATH")
    if seed_path and os.path.exists(seed_path):
        count = await load_training_data(seed_path)
        if count:
            logger.info(f"Seeded {count} training incidents into MongoDB")

    logger.info("API ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    logs: str = Field(..., min_length=10)
    metrics: Optional[str] = None
    error_trace: Optional[str] = None
    service: Optional[str] = None
    severity: Optional[str] = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        if v and v.lower() not in ("critical", "high", "medium", "low"):
            raise ValueError("severity must be critical, high, medium, or low")
        return v.lower() if v else v

    model_config = {"json_schema_extra": {"example": {
        "logs": "[2024-01-15 14:23:45] ERROR checkout-service: 248 queries in single request\n[2024-01-15 14:23:45] WARN ORM: lazy-loading orders table",
        "metrics": "DB CPU: 98%, Connection pool: 100%, P99 latency: 45000ms",
        "error_trace": "TimeoutError: request exceeded 45000ms\n  at checkout/handlers/checkout.py:142",
        "service": "checkout-service",
        "severity": "critical",
    }}}


class SimilarIncident(BaseModel):
    incident_id: str
    service: Optional[str] = None
    root_cause: Optional[str] = None
    timestamp: Optional[str] = None
    category: Optional[str] = None


class AnalyzeResponse(BaseModel):
    request_id: str
    analyzed_at: str
    root_cause: str
    confidence: float = Field(..., ge=0, le=1)
    category: str
    fix_steps: list[str]
    similar_incidents: list[SimilarIncident]
    model_used: str
    inference_time_ms: int


class FeedbackRequest(BaseModel):
    score: int = Field(..., ge=1, le=5)
    correct: bool
    comment: Optional[str] = Field(None, max_length=1000)


class GrafanaAlert(BaseModel):
    title: Optional[str] = None
    message: Optional[str] = None
    ruleName: Optional[str] = None
    state: Optional[str] = None
    tags: Optional[dict] = None
    evalMatches: Optional[list[dict]] = None
    logs: Optional[str] = None
    metrics: Optional[str] = None
    error_trace: Optional[str] = None
    service: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def notify_slack(result: dict, service: str, request_id: str):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    confidence_pct = int(result["confidence"] * 100)
    steps = "\n".join(f"• {s}" for s in result["fix_steps"][:3])
    payload = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"🚨 Incident Analysis: {service}"}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Root Cause:*\n{result['root_cause']}"},
                {"type": "mrkdwn", "text": f"*Confidence:* {confidence_pct}%\n*Category:* {result['category']}"},
            ]},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Fix Steps:*\n{steps}"}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Request: `{request_id}` | {result.get('inference_time_ms')}ms"}]},
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(webhook_url, json=payload)
    except Exception as exc:
        logger.warning(f"Slack notification failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        await db.command("ping")
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "model_type": os.environ.get("MODEL_TYPE", "claude"),
        "version": "1.0.0",
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_incident(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] service={req.service} severity={req.severity}")

    analyzer = get_analyzer()
    result = await analyzer.analyze(
        logs=req.logs or "",
        metrics=req.metrics or "",
        error_trace=req.error_trace or "",
        service=req.service or "unknown",
        db=db,
    )

    doc = {
        "request_id": request_id,
        "analyzed_at": datetime.utcnow(),
        "service": req.service,
        "severity": req.severity,
        "logs": req.logs,
        "metrics": req.metrics,
        "error_trace": req.error_trace,
        "root_cause": result["root_cause"],
        "confidence": result["confidence"],
        "fix_steps": result["fix_steps"],
        "category": result["category"],
        "model_used": result["model_used"],
        "similar_incident_ids": [s["incident_id"] for s in result.get("similar_incidents", [])],
        "inference_time_ms": result.get("inference_time_ms"),
    }
    await save_analysis(db, doc)

    logger.info(
        f"[{request_id}] category={result['category']} "
        f"confidence={result['confidence']:.0%} time={result.get('inference_time_ms')}ms"
    )

    if os.environ.get("SLACK_WEBHOOK_URL"):
        background_tasks.add_task(notify_slack, result, req.service or "unknown", request_id)

    return AnalyzeResponse(
        request_id=request_id,
        analyzed_at=doc["analyzed_at"].isoformat(),
        root_cause=result["root_cause"],
        confidence=result["confidence"],
        category=result["category"],
        fix_steps=result["fix_steps"],
        similar_incidents=[SimilarIncident(**s) for s in result.get("similar_incidents", [])],
        model_used=result["model_used"],
        inference_time_ms=result.get("inference_time_ms", 0),
    )


@app.post("/grafana/webhook", tags=["Integrations"])
async def grafana_webhook(
    alert: GrafanaAlert,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    eval_parts = [
        f"{m.get('metric', 'metric')}: {m.get('value', '?')}"
        for m in (alert.evalMatches or [])
    ]
    logs = "\n".join(filter(None, [
        f"Alert: {alert.title or alert.ruleName}",
        f"Message: {alert.message}" if alert.message else None,
        f"State: {alert.state}" if alert.state else None,
        f"Metrics: {', '.join(eval_parts)}" if eval_parts else None,
        alert.logs,
    ]))
    req = AnalyzeRequest(
        logs=logs,
        metrics=alert.metrics,
        error_trace=alert.error_trace,
        service=alert.service or (alert.tags or {}).get("service"),
        severity="critical" if alert.state == "alerting" else "medium",
    )
    return await analyze_incident(req, background_tasks, db)


@app.get("/incidents", tags=["Analysis"])
async def list_incidents(
    service: Optional[str] = None,
    category: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    total, items = await list_analyses(db, service, category, page, page_size)
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.get("/incidents/{request_id}", tags=["Analysis"])
async def get_incident(request_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    doc = await get_analysis(db, request_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Analysis {request_id} not found")
    return doc


@app.post("/feedback/{request_id}", tags=["Analysis"])
async def submit_feedback(
    request_id: str,
    feedback: FeedbackRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    found = await update_feedback(db, request_id, feedback.score, feedback.correct, feedback.comment)
    if not found:
        raise HTTPException(status_code=404, detail=f"Analysis {request_id} not found")
    return {"status": "ok", "message": "Feedback recorded. Thank you!"}


@app.get("/stats", tags=["System"])
async def stats(db: AsyncIOMotorDatabase = Depends(get_db)):
    return await get_stats(db)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("DEBUG", "false").lower() == "true",
        workers=int(os.environ.get("WORKERS", 1)),
    )

"""
Database layer using MongoDB (Motor async driver).
Collections:
  - incident_analyses   : every analysis request/response (audit trail)
  - training_incidents  : historical incident data for similarity search
"""

import json
import os
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB   = os.environ.get("MONGO_DB", "incident_analyzer")
USE_INMEMORY_DB = os.environ.get("USE_INMEMORY_DB", "0").lower() in ("1", "true", "yes")

# Local fallback store for environments without MongoDB (e.g., quick local demo).
_training_incidents_mem: list[dict] = []
_incident_analyses_mem: list[dict] = []

# ─── Client singleton ────────────────────────────────────────────────────────

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_client()[MONGO_DB]


# FastAPI dependency — yields the db for use in route handlers
async def get_db() -> AsyncIOMotorDatabase:
    if USE_INMEMORY_DB:
        yield None
        return
    yield get_database()


# ─── Init (create indexes) ───────────────────────────────────────────────────

async def init_db():
    if USE_INMEMORY_DB:
        return

    db = get_database()

    # incident_analyses indexes
    await db.incident_analyses.create_index("request_id", unique=True)
    await db.incident_analyses.create_index("analyzed_at")
    await db.incident_analyses.create_index("service")
    await db.incident_analyses.create_index("category")

    # training_incidents indexes
    await db.training_incidents.create_index("incident_id", unique=True)
    await db.training_incidents.create_index("category")
    await db.training_incidents.create_index("service")


# ─── Training data loader ─────────────────────────────────────────────────────

async def load_training_data(jsonl_path: str) -> int:
    if USE_INMEMORY_DB:
        count = 0
        existing_ids = {doc.get("incident_id") for doc in _training_incidents_mem}

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                incident_id = data.get("incident_id", f"INC-{len(_training_incidents_mem) + count:05d}")
                if incident_id in existing_ids:
                    continue

                ts_str = data.get("timestamp")
                ts = None
                if ts_str:
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        pass

                _training_incidents_mem.append({
                    "incident_id": incident_id,
                    "timestamp": ts,
                    "service": data.get("service"),
                    "severity": data.get("severity"),
                    "category": data.get("category"),
                    "logs": data.get("logs"),
                    "metrics": data.get("metrics"),
                    "error_trace": data.get("error_trace"),
                    "root_cause": data.get("root_cause"),
                    "resolution_steps": data.get("resolution_steps", []),
                })
                existing_ids.add(incident_id)
                count += 1

        return count

    db = get_database()
    count = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            incident_id = data.get("incident_id", f"INC-{count:05d}")

            # Skip if already exists
            existing = await db.training_incidents.find_one({"incident_id": incident_id})
            if existing:
                continue

            ts_str = data.get("timestamp")
            ts = None
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    pass

            doc = {
                "incident_id": incident_id,
                "timestamp": ts,
                "service": data.get("service"),
                "severity": data.get("severity"),
                "category": data.get("category"),
                "logs": data.get("logs"),
                "metrics": data.get("metrics"),
                "error_trace": data.get("error_trace"),
                "root_cause": data.get("root_cause"),
                "resolution_steps": data.get("resolution_steps", []),
            }
            await db.training_incidents.insert_one(doc)
            count += 1

    return count


# ─── Similarity search ────────────────────────────────────────────────────────

async def find_similar_incidents(
    db: AsyncIOMotorDatabase,
    category: str,
    service: str,
    limit: int = 3,
) -> list[dict]:
    if USE_INMEMORY_DB:
        same_service = [
            {
                "incident_id": d.get("incident_id"),
                "service": d.get("service"),
                "root_cause": d.get("root_cause"),
                "timestamp": d.get("timestamp").isoformat() if isinstance(d.get("timestamp"), datetime) else d.get("timestamp"),
                "category": d.get("category"),
            }
            for d in _training_incidents_mem
            if d.get("category") == category and d.get("service") == service
        ][:limit]

        seen = {d.get("incident_id") for d in same_service}
        if len(same_service) < limit:
            fill = [
                {
                    "incident_id": d.get("incident_id"),
                    "service": d.get("service"),
                    "root_cause": d.get("root_cause"),
                    "timestamp": d.get("timestamp").isoformat() if isinstance(d.get("timestamp"), datetime) else d.get("timestamp"),
                    "category": d.get("category"),
                }
                for d in _training_incidents_mem
                if d.get("category") == category and d.get("incident_id") not in seen
            ][: limit - len(same_service)]
            same_service.extend(fill)

        return same_service

    results = []

    # Same category + same service first
    async for doc in db.training_incidents.find(
        {"category": category, "service": service},
        {"_id": 0, "incident_id": 1, "service": 1, "root_cause": 1, "timestamp": 1, "category": 1},
        limit=limit,
    ):
        results.append(doc)

    # Fill remainder with same category only
    if len(results) < limit:
        seen = {r["incident_id"] for r in results}
        async for doc in db.training_incidents.find(
            {"category": category, "incident_id": {"$nin": list(seen)}},
            {"_id": 0, "incident_id": 1, "service": 1, "root_cause": 1, "timestamp": 1, "category": 1},
            limit=limit - len(results),
        ):
            results.append(doc)

    # Serialize datetimes
    for r in results:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()

    return results


# ─── Analysis CRUD ────────────────────────────────────────────────────────────

async def save_analysis(db: AsyncIOMotorDatabase, doc: dict) -> dict:
    if USE_INMEMORY_DB:
        _incident_analyses_mem.append(dict(doc))
        return doc

    await db.incident_analyses.insert_one(doc)
    doc.pop("_id", None)
    return doc


async def get_analysis(db: AsyncIOMotorDatabase, request_id: str) -> Optional[dict]:
    if USE_INMEMORY_DB:
        for doc in _incident_analyses_mem:
            if doc.get("request_id") == request_id:
                out = dict(doc)
                if isinstance(out.get("analyzed_at"), datetime):
                    out["analyzed_at"] = out["analyzed_at"].isoformat()
                return out
        return None

    doc = await db.incident_analyses.find_one(
        {"request_id": request_id}, {"_id": 0}
    )
    if doc and isinstance(doc.get("analyzed_at"), datetime):
        doc["analyzed_at"] = doc["analyzed_at"].isoformat()
    return doc


async def list_analyses(
    db: AsyncIOMotorDatabase,
    service: Optional[str] = None,
    category: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[int, list[dict]]:
    if USE_INMEMORY_DB:
        filtered = _incident_analyses_mem
        if service:
            filtered = [d for d in filtered if d.get("service") == service]
        if category:
            filtered = [d for d in filtered if d.get("category") == category]

        sorted_docs = sorted(filtered, key=lambda d: d.get("analyzed_at") or datetime.min, reverse=True)
        page_docs = sorted_docs[(page - 1) * page_size: page * page_size]

        docs = []
        for d in page_docs:
            item = {
                "request_id": d.get("request_id"),
                "analyzed_at": d.get("analyzed_at"),
                "service": d.get("service"),
                "severity": d.get("severity"),
                "root_cause": d.get("root_cause"),
                "confidence": d.get("confidence"),
                "category": d.get("category"),
                "inference_time_ms": d.get("inference_time_ms"),
            }
            if isinstance(item.get("analyzed_at"), datetime):
                item["analyzed_at"] = item["analyzed_at"].isoformat()
            docs.append(item)

        return len(filtered), docs

    filt = {}
    if service:
        filt["service"] = service
    if category:
        filt["category"] = category

    total = await db.incident_analyses.count_documents(filt)
    docs = []
    async for doc in db.incident_analyses.find(
        filt,
        {"_id": 0, "request_id": 1, "analyzed_at": 1, "service": 1,
         "severity": 1, "root_cause": 1, "confidence": 1,
         "category": 1, "inference_time_ms": 1},
        sort=[("analyzed_at", -1)],
        skip=(page - 1) * page_size,
        limit=page_size,
    ):
        if isinstance(doc.get("analyzed_at"), datetime):
            doc["analyzed_at"] = doc["analyzed_at"].isoformat()
        docs.append(doc)

    return total, docs


async def update_feedback(
    db: AsyncIOMotorDatabase,
    request_id: str,
    score: int,
    correct: bool,
    comment: Optional[str],
) -> bool:
    if USE_INMEMORY_DB:
        for doc in _incident_analyses_mem:
            if doc.get("request_id") == request_id:
                doc["feedback_score"] = score
                doc["feedback_correct"] = 1 if correct else 0
                doc["feedback_comment"] = comment
                return True
        return False

    result = await db.incident_analyses.update_one(
        {"request_id": request_id},
        {"$set": {
            "feedback_score": score,
            "feedback_correct": 1 if correct else 0,
            "feedback_comment": comment,
        }},
    )
    return result.matched_count > 0


async def get_stats(db: AsyncIOMotorDatabase) -> dict:
    if USE_INMEMORY_DB:
        total = len(_incident_analyses_mem)
        avg_conf = sum(float(d.get("confidence") or 0) for d in _incident_analyses_mem) / total if total else 0
        avg_ms = sum(float(d.get("inference_time_ms") or 0) for d in _incident_analyses_mem) / total if total else 0

        cat_dist: dict[str, int] = {}
        for d in _incident_analyses_mem:
            cat = d.get("category")
            if cat:
                cat_dist[cat] = cat_dist.get(cat, 0) + 1

        correct = sum(1 for d in _incident_analyses_mem if d.get("feedback_correct") == 1)
        total_fb = sum(1 for d in _incident_analyses_mem if "feedback_correct" in d)

        return {
            "total_analyses": total,
            "avg_confidence": round(avg_conf, 3),
            "avg_inference_ms": round(avg_ms),
            "feedback_accuracy": round(correct / total_fb, 3) if total_fb else None,
            "feedback_count": total_fb,
            "category_distribution": cat_dist,
        }

    pipeline = [
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence"},
            "avg_inference_ms": {"$avg": "$inference_time_ms"},
        }}
    ]
    agg = await db.incident_analyses.aggregate(pipeline).to_list(1)
    base = agg[0] if agg else {"total": 0, "avg_confidence": 0, "avg_inference_ms": 0}

    # Category distribution
    cat_pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    cat_dist = {
        doc["_id"]: doc["count"]
        async for doc in db.incident_analyses.aggregate(cat_pipeline)
        if doc["_id"]
    }

    # Feedback accuracy
    correct = await db.incident_analyses.count_documents({"feedback_correct": 1})
    total_fb = await db.incident_analyses.count_documents(
        {"feedback_correct": {"$exists": True}}
    )

    return {
        "total_analyses": base.get("total", 0),
        "avg_confidence": round(float(base.get("avg_confidence") or 0), 3),
        "avg_inference_ms": round(float(base.get("avg_inference_ms") or 0)),
        "feedback_accuracy": round(correct / total_fb, 3) if total_fb else None,
        "feedback_count": total_fb,
        "category_distribution": cat_dist,
    }

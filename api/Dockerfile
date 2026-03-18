##############################################################################
# Smart Incident Root Cause Analyzer — Production Dockerfile
# Supports both MODEL_TYPE=claude (lightweight) and MODEL_TYPE=fine_tuned (GPU)
##############################################################################

FROM python:3.11-slim AS base

LABEL maintainer="incident-analyzer"
LABEL description="AI-powered production incident root cause analysis API"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ─── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .

# Install API/runtime dependencies used in claude/openai mode.
RUN pip install \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    motor \
    httpx \
    anthropic \
    openai \
    python-dotenv \
    boto3

# Install ML deps conditionally based on MODEL_TYPE arg
ARG MODEL_TYPE=claude
RUN if [ "$MODEL_TYPE" = "fine_tuned" ]; then \
    pip install torch transformers peft accelerate bitsandbytes sentencepiece; \
    fi

# ─── Application code ────────────────────────────────────────────────────────
COPY . .

# ─── Non-root user ───────────────────────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# ─── Health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE ${PORT:-8000}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]

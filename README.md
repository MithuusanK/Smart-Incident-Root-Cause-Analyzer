# Smart Incident Root Cause Analyzer

**AI-powered production incident analysis.** Fine-tune a Mistral-7B model on historical incident data, then deploy it as a real-time API that identifies root causes in minutes — not hours.

```
Incident fires → Agent analyzes logs → 2 minutes:
"Root cause: N+1 query in checkout service.
 Confidence: 92%. Fix: add select_related().
 Similar incidents: INC-00234, INC-00891"
```

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Grafana Alert   │───▶│  FastAPI /analyze │───▶│  Slack / Webhook │
│  or Web UI       │    │  (DigitalOcean)   │    │  Notification    │
└──────────────────┘    └────────┬─────────┘    └──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Fine-tuned Mistral-7B   │
                    │  (QLoRA on DO GPU)       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  MongoDB                │
                    │  (incident history +    │
                    │   similarity search)    │
                    └─────────────────────────┘
```

**Stack:**
- **GPU Training:** DigitalOcean GPU Droplet (A100 40GB)
- **Base Model:** Mistral-7B-v0.1 (HuggingFace)
- **Fine-tuning:** QLoRA via PEFT + trl (4-bit, ~15GB VRAM)
- **Inference API:** FastAPI on DigitalOcean App Platform
- **Database:** MongoDB (DigitalOcean Managed MongoDB or local Mongo)
- **Demo mode:** Claude API (claude-haiku) when GPU model not deployed

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/your-org/incident-root-cause-analyzer
cd incident-root-cause-analyzer
cp api/.env.example api/.env
# Edit api/.env — add ANTHROPIC_API_KEY for demo mode
```

### 2. Generate training data

```bash
cd data
python -m pip install -r ../train/requirements.txt
python generate_synthetic_data.py --count 900
# → data/training_incidents.jsonl      (765 examples)
# → data/training_incidents_test.jsonl (135 examples)
```

### 3. Run full stack locally

```bash
# Requires: Docker + Docker Compose
ANTHROPIC_API_KEY=sk-ant-... docker compose up -d

# API:       http://localhost:8000
# Dashboard: http://localhost:3000
# API Docs:  http://localhost:8000/docs
```

### 4. Test the API

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "service": "checkout-service",
    "severity": "critical",
    "logs": "[2024-01-15 14:23:45] ERROR: 248 queries in single request\n[2024-01-15 14:23:45] WARN: ORM lazy-loading orders table\n[2024-01-15 14:23:45] ERROR: Request timeout 45000ms",
    "metrics": "DB CPU: 98%, Connection pool: 100%, P99: 45000ms"
  }'
```

**Response:**
```json
{
  "request_id": "a1b2c3d4-...",
  "analyzed_at": "2024-01-15T14:23:46Z",
  "root_cause": "N+1 query problem in checkout-service: ORM lazy-loading orders records individually (248 queries instead of 1 JOIN)",
  "confidence": 0.92,
  "category": "database_n_plus_one",
  "fix_steps": [
    "Replace lazy-loading with eager JOIN: queryset.select_related('orders')",
    "Add database query count assertion in integration tests",
    "Set ORM strict mode to raise on N+1 in non-production environments"
  ],
  "similar_incidents": [
    {"incident_id": "INC-00234", "service": "checkout-service", "root_cause": "N+1 query..."}
  ],
  "model_used": "claude",
  "inference_time_ms": 1243
}
```

---

## Phase 1: Data Preparation

### Dataset Schema

```json
{
  "incident_id": "INC-00001",
  "timestamp": "2024-01-15T14:23:45Z",
  "service": "checkout-service",
  "severity": "critical",
  "logs": "...",
  "metrics": "DB CPU: 98%, ...",
  "error_trace": "TimeoutError: ...",
  "root_cause": "N+1 query in checkout service",
  "resolution_steps": ["Step 1", "Step 2"],
  "category": "database_n_plus_one",
  "instruction": "Analyze this production incident...",
  "input": "SERVICE: checkout-service\nLOGS:\n...",
  "output": "ROOT CAUSE: ...\nCONFIDENCE: 92%\nRESOLUTION STEPS:\n1. ..."
}
```

### Incident Categories (15 types)

| Category | Description |
|---|---|
| `database_n_plus_one` | ORM lazy-loading N rows with N queries |
| `database_connection_pool` | Pool exhausted under load |
| `database_missing_index` | Full table scan on large table |
| `database_deadlock` | Inconsistent lock acquisition order |
| `memory_leak` | Unbounded cache / no GC eviction |
| `kubernetes_oom` | Container exceeds memory limit |
| `cpu_hot_loop` | Regex backtracking / infinite loop |
| `disk_full` | Log files / WAL filling partition |
| `cache_stampede` | TTL expiry under concurrent load |
| `service_timeout_cascade` | Downstream timeout propagates up |
| `certificate_expiry` | TLS cert expired, HTTPS fails |
| `rate_limit_breach` | External API rate limit hit |
| `config_error` | Missing env var on deployment |
| `network_partition` | AZ unreachable, quorum lost |
| `application_error` | Breaking schema change / exception |

---

## Phase 2: Fine-tuning on DigitalOcean GPU

### GPU Droplet Setup

```bash
# Create GPU droplet via DO CLI
doctl compute droplet create incident-trainer \
  --image gpu-h100x1-base \
  --size gpu-h100x1-80gb \
  --region nyc3 \
  --ssh-keys YOUR_KEY_ID

# SSH in
ssh root@YOUR_DROPLET_IP

# Install dependencies
pip install -r train/requirements.txt
```

### Fine-tune Command

```bash
python train/train.py \
  --model_name mistralai/Mistral-7B-v0.1 \
  --train_file data/training_incidents.jsonl \
  --eval_file data/training_incidents_test.jsonl \
  --output_dir ./train/checkpoints/incident-analyzer-v1 \
  --num_epochs 5 \
  --learning_rate 2e-4 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --lora_r 64 \
  --lora_alpha 128 \
  --spaces_bucket YOUR_BUCKET_NAME   # uploads to DO Spaces
```

### Training Config (QLoRA)

| Parameter | Value |
|---|---|
| Base model | Mistral-7B-v0.1 |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| Target modules | q,k,v,o,gate,up,down |
| Epochs | 5 |
| Learning rate | 2e-4 (cosine decay) |
| Batch size | 4 × 4 grad accum = 16 eff. |
| VRAM required | ~15 GB (fits H100 / A100) |

### Evaluate

```bash
python train/eval.py \
  --model_dir ./train/checkpoints/incident-analyzer-v1/final \
  --test_file data/training_incidents_test.jsonl \
  --output_file train/eval_results.json
```

Target: **>80% category accuracy**, **ROUGE-L >0.60**

---

## Phase 3: API Deployment

### Environment Variables

```bash
# Required
MONGO_URL=mongodb+srv://...
MONGO_DB=incident_analyzer
ANTHROPIC_API_KEY=sk-ant-...         # for demo/claude mode

# For fine-tuned model
MODEL_TYPE=fine_tuned
MODEL_CHECKPOINT=./checkpoints/final

# Integrations
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

### DigitalOcean App Platform Deployment

```yaml
# app.yaml (DO App Platform spec)
name: incident-analyzer
services:
  - name: api
    source_dir: api
    dockerfile_path: api/Dockerfile
    instance_size_slug: apps-d-4vcpu-8gb   # or GPU instance
    instance_count: 2
    http_port: 8000
    envs:
      - key: MONGO_URL
        value: ${incident-analyzer-db.DATABASE_URL}
      - key: MONGO_DB
        value: incident_analyzer
      - key: MODEL_TYPE
        value: claude
      - key: ANTHROPIC_API_KEY
        type: SECRET
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Analyze an incident |
| `GET` | `/health` | Health check |
| `GET` | `/incidents` | List analyses (paginated) |
| `GET` | `/incidents/{id}` | Get analysis details |
| `POST` | `/feedback/{id}` | Submit correctness feedback |
| `POST` | `/grafana/webhook` | Grafana alert webhook |
| `GET` | `/stats` | Aggregate statistics |
| `GET` | `/docs` | Interactive API docs (Swagger) |

---

## Phase 4: Integrations

### Grafana Webhook

Configure Grafana alert contact point:
- **Type:** Webhook
- **URL:** `https://your-api.ondigitalocean.app/grafana/webhook`
- **HTTP Method:** POST

Test scenarios:
```bash
cd integrations
pip install httpx
python grafana_webhook.py --api-url http://localhost:8000 --list-scenarios
python grafana_webhook.py --api-url http://localhost:8000 --scenario db_cpu
python grafana_webhook.py --api-url http://localhost:8000 --all
```

### Slack Bot

```bash
pip install slack-bolt flask
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...
export ANALYZER_API_URL=http://localhost:8000
python integrations/slack_bot.py

# Test without Slack:
python integrations/slack_bot.py --test
```

Slash command: `/analyze-incident [paste logs here]`

---

## Repository Structure

```
incident-root-cause-analyzer/
├── README.md
├── docker-compose.yml          # Full stack: API + MongoDB + Dashboard
├── nginx.conf                  # Dashboard reverse proxy
├── data/
│   ├── generate_synthetic_data.py   # Generates 900 training examples
│   └── training_incidents.jsonl     # Generated dataset (run script first)
├── train/
│   ├── train.py                # QLoRA fine-tuning (Mistral-7B)
│   ├── eval.py                 # Evaluation: accuracy, F1, ROUGE
│   └── requirements.txt
├── api/
│   ├── main.py                 # FastAPI app (all endpoints)
│   ├── models.py               # Inference: fine-tuned model + Claude fallback
│   ├── database.py             # Motor/MongoDB data access layer
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── dashboard/
│   ├── index.html              # Single-page web UI
│   └── style.css
├── integrations/
│   ├── grafana_webhook.py      # Grafana webhook test utility
│   └── slack_bot.py            # Slack bot (slash command + mentions)
└── notebooks/
    └── training_analysis.ipynb # Dataset EDA, loss curves, eval charts
```

---

## Success Checklist

- [ ] `python data/generate_synthetic_data.py` generates 900 examples
- [ ] `docker compose up` starts API at `:8000`, dashboard at `:3000`
- [ ] `POST /analyze` responds in <5 seconds (Claude mode) or <30s (fine-tuned)
- [ ] Web UI: paste logs → get root cause + confidence + fix steps
- [ ] Fine-tuned model achieves >80% category accuracy on test set
- [ ] Grafana webhook sends alert → `/grafana/webhook` → Slack notification
- [ ] All analyses logged to MongoDB with audit trail

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built for the DigitalOcean AI/ML Hackathon. Incident response, 10x faster.*

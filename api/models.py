"""
Inference logic for the Incident Root Cause Analyzer.
Supports two backends:
  1. FineTunedModel  — loads QLoRA-adapted Mistral-7B from local checkpoint
  2. ClaudeModel     — uses Anthropic Claude API (demo / fallback)

Set MODEL_TYPE=fine_tuned or MODEL_TYPE=claude in .env
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

MODEL_TYPE       = os.environ.get("MODEL_TYPE", "claude")
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "./checkpoints/incident-analyzer-v1/final")

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) and production incident analyst.
Analyze the production incident from the provided logs, metrics, and error traces, then identify:
1. The precise root cause
2. A confidence score (0-100)
3. Step-by-step resolution actions
4. The incident category

Always respond in this exact JSON format:
{
  "root_cause": "<concise description of the root cause>",
  "confidence": <integer 0-100>,
  "category": "<one of: database_n_plus_one, database_connection_pool, memory_leak, cpu_hot_loop, disk_full, database_missing_index, kubernetes_oom, cache_stampede, database_deadlock, service_timeout_cascade, certificate_expiry, rate_limit_breach, config_error, network_partition, application_error>",
  "fix_steps": ["<step 1>", "<step 2>", "<step 3>"]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseIncidentModel(ABC):
    @abstractmethod
    async def analyze(self, logs: str, metrics: str, error_trace: str, service: str) -> dict[str, Any]:
        ...

    def preprocess(self, logs: str, metrics: str, error_trace: str, service: str) -> str:
        parts = [f"SERVICE: {service or 'unknown'}"]
        if logs.strip():
            parts.append(f"\nLOGS:\n{logs.strip()}")
        if metrics.strip():
            parts.append(f"\nMETRICS:\n{metrics.strip()}")
        if error_trace.strip():
            parts.append(f"\nERROR TRACE:\n{error_trace.strip()}")
        return "\n".join(parts)

    def parse_json_output(self, text: str) -> dict[str, Any]:
        # Direct JSON parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # JSON block inside text
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Plain text fallback
        root_cause = ""
        confidence = 75
        fix_steps = []
        rc = re.search(r"ROOT CAUSE[:\s]+(.+?)(?:\n|CONFIDENCE|FIX)", text, re.IGNORECASE | re.DOTALL)
        if rc:
            root_cause = rc.group(1).strip()
        cf = re.search(r"CONFIDENCE[:\s]+(\d+)", text, re.IGNORECASE)
        if cf:
            confidence = int(cf.group(1))
        fix_steps = re.findall(r"^\d+\.\s+(.+)$", text, re.MULTILINE)
        return {
            "root_cause": root_cause or text[:200],
            "confidence": confidence,
            "category": "unknown",
            "fix_steps": fix_steps or ["Investigate logs further", "Check recent deployments"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuned Mistral backend
# ─────────────────────────────────────────────────────────────────────────────

class FineTunedModel(BaseIncidentModel):
    def __init__(self, checkpoint_dir: str):
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        logger.info(f"Loading fine-tuned model from {checkpoint_dir}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Fine-tuned model loaded")

    async def analyze(self, logs: str, metrics: str, error_trace: str, service: str) -> dict[str, Any]:
        import asyncio
        import torch

        incident_text = self.preprocess(logs, metrics, error_trace, service)
        prompt = (
            f"### Instruction:\nAnalyze this production incident and return a JSON response.\n\n"
            f"### Input:\n{incident_text}\n\n"
            f"### Response:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(self.device)

        def _generate():
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=400, temperature=0.1,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Run blocking inference in thread pool so FastAPI stays responsive
        text = await asyncio.get_event_loop().run_in_executor(None, _generate)
        return self.parse_json_output(text)


# ─────────────────────────────────────────────────────────────────────────────
# Claude API backend
# ─────────────────────────────────────────────────────────────────────────────

class ClaudeModel(BaseIncidentModel):
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_id = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        logger.info(f"Using Claude API: {self.model_id}")

    async def analyze(self, logs: str, metrics: str, error_trace: str, service: str) -> dict[str, Any]:
        import asyncio
        incident_text = self.preprocess(logs, metrics, error_trace, service)

        def _call():
            message = self.client.messages.create(
                model=self.model_id,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Analyze this production incident:\n\n{incident_text}"}],
            )
            return message.content[0].text

        text = await asyncio.get_event_loop().run_in_executor(None, _call)
        return self.parse_json_output(text)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI GPT backend
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIModel(BaseIncidentModel):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"Using OpenAI API: {self.model_id}")

    async def analyze(self, logs: str, metrics: str, error_trace: str, service: str) -> dict[str, Any]:
        import asyncio
        incident_text = self.preprocess(logs, metrics, error_trace, service)

        def _call():
            response = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this production incident:\n\n{incident_text}"},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        text = await asyncio.get_event_loop().run_in_executor(None, _call)
        return self.parse_json_output(text)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IncidentAnalyzer:
    def __init__(self):
        self._model: BaseIncidentModel | None = None

    def _load_model(self) -> BaseIncidentModel:
        if self._model is not None:
            return self._model
        if MODEL_TYPE.lower() == "fine_tuned":
            self._model = FineTunedModel(MODEL_CHECKPOINT)
        elif MODEL_TYPE.lower() == "claude":
            self._model = ClaudeModel()
        elif MODEL_TYPE.lower() == "openai":
            self._model = OpenAIModel()
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'claude', 'openai', or 'fine_tuned'")
        return self._model

    async def analyze(self, logs: str, metrics: str, error_trace: str, service: str, db) -> dict[str, Any]:
        from database import find_similar_incidents

        model = self._load_model()
        start = time.time()
        result = await model.analyze(logs, metrics, error_trace, service)
        inference_ms = int((time.time() - start) * 1000)

        confidence = result.get("confidence", 75)
        if confidence > 1:
            confidence = confidence / 100.0
        confidence = round(min(max(float(confidence), 0.0), 1.0), 3)

        category = result.get("category", "unknown")
        similar = await find_similar_incidents(db, category, service)

        return {
            "root_cause": result.get("root_cause", "Unable to determine root cause"),
            "confidence": confidence,
            "category": category,
            "fix_steps": result.get("fix_steps", []),
            "similar_incidents": similar,
            "model_used": MODEL_TYPE,
            "inference_time_ms": inference_ms,
        }


_analyzer: IncidentAnalyzer | None = None


def get_analyzer() -> IncidentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = IncidentAnalyzer()
    return _analyzer

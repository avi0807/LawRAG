"""
Structured logging + per-query trace recording.

- `log` is a stdlib logger configured for JSON-ish output.
- `Trace` collects per-stage timings and a small payload per stage,
  flushed to data/logs/traces-YYYY-MM-DD.jsonl on completion.

Trace files are append-only JSONL, one line per query. Cheap to grep, cheap to
load into pandas for analysis, and they survive process restarts.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from config import cfg


# ──────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Surface anything the caller attached as `extra={"foo": ...}`
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName", "taskName",
            ):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        return json.dumps(payload, ensure_ascii=False)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("lawrag")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
    logger.propagate = False

    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)
    return logger


log = _build_logger()


# ──────────────────────────────────────────────────────────
# Per-query trace
# ──────────────────────────────────────────────────────────

@dataclass
class StageRecord:
    stage: str
    started_at: float
    duration_ms: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    query: str
    session_id: str = "default"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    stages: List[StageRecord] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str, **payload: Any) -> Iterator["StageRecord"]:
        start = time.time()
        rec = StageRecord(stage=name, started_at=start, duration_ms=0.0, payload=dict(payload))
        try:
            yield rec
        finally:
            rec.duration_ms = round((time.time() - start) * 1000, 2)
            self.stages.append(rec)
            log.info(
                f"stage.{name}",
                extra={
                    "trace_id": self.trace_id,
                    "stage": name,
                    "duration_ms": rec.duration_ms,
                    **rec.payload,
                },
            )

    def total_ms(self) -> float:
        return round((time.time() - self.started_at) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "started_at": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "total_ms": self.total_ms(),
            "stages": [
                {
                    "stage": s.stage,
                    "duration_ms": s.duration_ms,
                    **s.payload,
                }
                for s in self.stages
            ],
            "final": self.final,
        }

    def flush(self) -> None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = os.path.join(cfg.log_dir, f"traces-{day}.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:  # never let logging break the request
            log.warning(f"trace.flush.failed: {e}")

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import Agent, TokenUsage, UsageEvent, parse_timestamp
from .base import Adapter, nested, nonnegative_int, stable_id


class PiFamilyAdapter(Adapter):
    """pi and omp share one session JSONL format; only paths and a few field names differ."""

    agent: Agent
    sessions_dir: str = ""

    @classmethod
    def default_paths(cls) -> list[Path]:
        return [Path(f"~/{cls.sessions_dir}")]

    def collect(self, since: datetime | None = None) -> list[UsageEvent]:
        events: list[UsageEvent] = []
        for path in self.files(("*.jsonl",)):
            events.extend(self._collect_file(path, since))
        return self.finish(events)

    def _collect_file(self, path: Path, since: datetime | None) -> list[UsageEvent]:
        events: list[UsageEvent] = []
        session_id = path.stem
        project = ""
        fallback_model = ""
        for data in self.read_jsonl(path):
            kind = data.get("type")
            if kind == "session":
                session_id = str(data.get("id") or session_id)
                project = str(data.get("cwd") or project)
            elif kind == "model_change":
                fallback_model = _model_name(data)
            elif kind == "message":
                message = data.get("message") if isinstance(data.get("message"), dict) else {}
                if message.get("role") != "assistant":
                    continue
                usage = message.get("usage")
                if not isinstance(usage, dict):
                    continue
                timestamp = parse_timestamp(data.get("timestamp") or message.get("timestamp"))
                if timestamp is None:
                    continue
                event = UsageEvent(
                    id=stable_id(self.agent.value, path, data.get("id"), timestamp.isoformat()),
                    agent=self.agent,
                    timestamp=timestamp,
                    usage=TokenUsage(
                        input=nonnegative_int(usage.get("input")),
                        output=nonnegative_int(usage.get("output")),
                        cache_read=nonnegative_int(usage.get("cacheRead")),
                        cache_write=nonnegative_int(usage.get("cacheWrite")),
                        reasoning=nonnegative_int(
                            usage.get("reasoning") or usage.get("reasoningTokens")
                        ),
                    ),
                    model=_model_name(message) or fallback_model or "unknown",
                    session_id=session_id,
                    project=project or path.parent.name,
                    cost_usd=_optional_float(nested(usage, "cost", "total")),
                    source_kind="pi_session_jsonl",
                    source_path=str(path),
                )
                if since is None or event.timestamp >= since:
                    events.append(event)
        return events


class PiAdapter(PiFamilyAdapter):
    agent = Agent.PI
    sessions_dir = ".pi/agent/sessions"


class OmpAdapter(PiFamilyAdapter):
    agent = Agent.OMP
    sessions_dir = ".omp/agent/sessions"


def _model_name(source: dict[str, Any]) -> str:
    model = source.get("model") or source.get("modelId") or ""
    if not model:
        return ""
    provider = source.get("provider") or ""
    return f"{provider}/{model}" if provider else str(model)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

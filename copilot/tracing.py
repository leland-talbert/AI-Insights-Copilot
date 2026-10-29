"""OpenTelemetry tracing to Arize Phoenix using OpenInference conventions.

OpenAI calls (LLM + embeddings) are auto-instrumented. Retrieval and the
citation check get their own spans, so a trace shows retrieved passages,
the prompt, tokens, latency, and what the guardrail rejected.
If Phoenix is disabled or unreachable, tracing becomes a no-op.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from opentelemetry import trace

from . import config

log = logging.getLogger(__name__)
tracer = trace.get_tracer("ai-insights-copilot")
_state = {"initialized": False, "enabled": False}

SPAN_KIND = "openinference.span.kind"


def init_tracing() -> bool:
    if _state["initialized"]:
        return _state["enabled"]
    _state["initialized"] = True
    if not config.PHOENIX_ENABLED:
        return False
    try:
        from phoenix.otel import register

        register(
            project_name=config.PHOENIX_PROJECT,
            endpoint=f"{config.PHOENIX_ENDPOINT.rstrip('/')}/v1/traces",
            auto_instrument=True,
            batch=True,
            verbose=False,
        )
        _state["enabled"] = True
    except Exception as exc:  # pragma: no cover
        log.warning("Phoenix tracing disabled: %s", exc)
    return _state["enabled"]


def tracing_enabled() -> bool:
    return _state["enabled"]


def set_io(span, input_value: Any = None, output_value: Any = None) -> None:
    if input_value is not None:
        span.set_attribute("input.value", input_value if isinstance(input_value, str) else json.dumps(input_value, default=str))
    if output_value is not None:
        span.set_attribute("output.value", output_value if isinstance(output_value, str) else json.dumps(output_value, default=str))
        span.set_attribute("output.mime_type", "text/plain" if isinstance(output_value, str) else "application/json")


def record_retrieved_documents(span, docs: Sequence[dict]) -> None:
    """docs: dicts with id, content, score, metadata."""
    for i, doc in enumerate(docs):
        prefix = f"retrieval.documents.{i}.document"
        span.set_attribute(f"{prefix}.id", str(doc["id"]))
        span.set_attribute(f"{prefix}.content", doc["content"])
        span.set_attribute(f"{prefix}.score", float(doc["score"]))
        span.set_attribute(f"{prefix}.metadata", json.dumps(doc.get("metadata", {}), default=str))

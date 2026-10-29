"""Model provider layer.

OpenAI models use the Responses API with strict JSON-schema output.
Local models run through Ollama's OpenAI-compatible endpoint, so swapping
providers is a configuration change, not a code change.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache

from . import config
from .cost import Usage


class LLMError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def _ollama_client():
    from openai import OpenAI

    return OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")


def resolve_model(model_key: str) -> config.ModelOption:
    if model_key in config.MODELS:
        return config.MODELS[model_key]
    return config.ModelOption(model_key, "openai", model_key, model_key)


def repair_json(text: str) -> dict:
    """Parse model output as JSON, repairing common defects (fences, prose, trailing commas)."""
    if not text or not text.strip():
        raise LLMError("Model returned an empty response.")
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start:end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model output was not valid JSON: {exc}") from exc


def generate_json(
    *,
    model_key: str,
    instructions: str,
    user_input: str,
    schema: dict,
    schema_name: str,
    purpose: str = "generation",
    temperature: float = 0.0,
) -> tuple[dict, Usage]:
    option = resolve_model(model_key)
    start = time.perf_counter()
    try:
        if option.provider == "openai":
            resp = _openai_client().responses.create(
                model=option.model_id,
                instructions=instructions,
                input=user_input,
                temperature=temperature,
                text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
            )
            text = resp.output_text
            in_tok = getattr(resp.usage, "input_tokens", 0) or 0
            out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        else:
            resp = _ollama_client().chat.completions.create(
                model=option.model_id,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
                response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
            )
            text = resp.choices[0].message.content or ""
            in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
            out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"{option.label} request failed: {exc}") from exc

    usage = Usage(option.provider, option.model_id, in_tok, out_tok, time.perf_counter() - start, purpose)
    return repair_json(text), usage

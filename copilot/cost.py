"""Token usage and cost accounting, computed from the usage the API actually reports."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config


@dataclass
class Usage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    purpose: str = ""

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens, self.provider)


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0, provider: str = "openai") -> float:
    """Estimated USD cost. Local (Ollama) inference is reported as $0 API spend."""
    if provider == "ollama":
        return 0.0
    price_in, price_out = config.PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class CostLedger:
    """Running per-session tally."""

    entries: list[Usage] = field(default_factory=list)

    def add(self, usages: list[Usage]) -> None:
        self.entries.extend(usages)

    @property
    def total_usd(self) -> float:
        return sum(u.cost_usd for u in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self.entries)


def summarize(usages: list[Usage]) -> dict:
    return {
        "input_tokens": sum(u.input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "cost_usd": round(sum(u.cost_usd for u in usages), 6),
        "latency_s": round(sum(u.latency_s for u in usages), 3),
    }

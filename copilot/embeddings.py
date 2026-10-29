"""Embedding providers. OpenAI text-embedding-3-small by default."""

from __future__ import annotations

import time
from typing import Protocol, Sequence

import numpy as np

from . import config
from .cost import Usage


class Embedder(Protocol):
    model: str
    provider: str

    def embed(self, texts: Sequence[str]) -> tuple[list[np.ndarray], Usage]: ...


class OpenAIEmbedder:
    provider = "openai"

    def __init__(self, model: str | None = None, batch_size: int = 96):
        from openai import OpenAI

        self.model = model or config.EMBEDDING_MODEL
        self.batch_size = batch_size
        self._client = OpenAI()

    def embed(self, texts: Sequence[str]) -> tuple[list[np.ndarray], Usage]:
        start = time.perf_counter()
        vectors: list[np.ndarray] = []
        tokens = 0
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[i:i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(np.asarray(item.embedding, dtype=np.float32) for item in resp.data)
            tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
        usage = Usage(self.provider, self.model, tokens, 0, time.perf_counter() - start, "embedding")
        return vectors, usage

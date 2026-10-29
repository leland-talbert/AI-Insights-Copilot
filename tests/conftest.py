import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.cost import Usage  # noqa: E402


class HashEmbedder:
    """Offline, deterministic bag-of-words embedder for tests only (no API calls)."""

    provider = "test"
    model = "hash-embedder-test"
    dim = 1536

    def embed(self, texts):
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) < 3:
                    continue
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm else vec)
        return vectors, Usage("test", self.model, sum(len(t) // 4 for t in texts), 0, 0.0, "embedding")


@pytest.fixture
def embedder():
    return HashEmbedder()


def _db_available() -> bool:
    try:
        import psycopg

        from copilot import config

        with psycopg.connect(config.DATABASE_URL, connect_timeout=2):
            return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    os.getenv("SKIP_DB_TESTS") == "1" or not _db_available(),
    reason="Postgres with pgvector not reachable at DATABASE_URL",
)

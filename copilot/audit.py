"""Append-only, hash-chained review trail for human-in-the-loop decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from . import store

GENESIS_HASH = "0" * 64
_LOCK_KEY = 7_420_431  # advisory lock that serializes appends to the chain


def canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_hash(prev_hash: str, record: dict) -> str:
    return hashlib.sha256((prev_hash + canonical(record)).encode("utf-8")).hexdigest()


def record_event(workspace: str, reviewer: str, action: str, target: str, details: dict | None = None) -> dict:
    reviewer = (reviewer or "").strip() or "unidentified reviewer"
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace": workspace,
        "reviewer": reviewer,
        "action": action,
        "target": target,
        "details": details or {},
    }
    with store.connect() as conn, conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        last = conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = last["hash"] if last else GENESIS_HASH
        digest = compute_hash(prev_hash, record)
        row = conn.execute(
            """INSERT INTO audit_log (created_at, workspace, reviewer, action, target, record, prev_hash, hash)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s) RETURNING id""",
            (record["created_at"], workspace, reviewer, action, target, canonical(record), prev_hash, digest),
        ).fetchone()
    return {"id": row["id"], "hash": digest, **record}


def list_events(workspace: str | None = None, limit: int = 500) -> list[dict]:
    sql = "SELECT id, created_at, workspace, reviewer, action, target, record, prev_hash, hash FROM audit_log"
    params: list = []
    if workspace:
        sql += " WHERE workspace = %s"
        params.append(workspace)
    sql += " ORDER BY id DESC LIMIT %s"
    params.append(limit)
    with store.connect() as conn:
        return conn.execute(sql, params).fetchall()


def verify_chain() -> tuple[bool, int, str]:
    """Recompute every hash from the genesis value. Returns (ok, rows_checked, message)."""
    with store.connect() as conn:
        rows = conn.execute("SELECT id, record, prev_hash, hash FROM audit_log ORDER BY id").fetchall()
    expected_prev = GENESIS_HASH
    for row in rows:
        if row["prev_hash"] != expected_prev:
            return False, len(rows), f"Chain broken at entry {row['id']}: previous-hash link does not match."
        if compute_hash(row["prev_hash"], row["record"]) != row["hash"]:
            return False, len(rows), f"Entry {row['id']} was altered: its content no longer matches its hash."
        expected_prev = row["hash"]
    return True, len(rows), f"All {len(rows)} entries verified."

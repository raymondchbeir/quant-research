from __future__ import annotations

"""No-API regression tests for V1.5 bounded JSONL ingestion.

Run as a module or call ``run(show=True)``.  No exchange API is called and no
orders are sent.
"""

import json
import tempfile
from pathlib import Path

from . import mm_deep_tail_join_ask_live_v1_5 as V15


def run(*, show=True):
    checks = {}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tail.jsonl"
        rows = [{"i": i, "payload": "x" * 80} for i in range(1000)]
        with p.open("wb") as fh:
            for row in rows:
                fh.write(json.dumps(row).encode("utf-8") + b"\n")

        tail = V15.BoundedJsonlTail(p, max_rows=37, max_bytes=4096)
        got = []
        per_call = []
        for _ in range(1000):
            batch = tail.read_new()
            per_call.append(len(batch))
            got.extend(batch)
            if not batch and tail.backlog_bytes() == 0:
                break

        checks["all_rows_recovered_in_order"] = [r["i"] for r in got] == list(range(1000))
        checks["row_budget_respected"] = max(per_call or [0]) <= 37
        checks["bounded_reader_reaches_zero_backlog"] = tail.backlog_bytes() == 0
        checks["no_decode_errors"] = tail.decode_errors == 0

        # Partial writer-tail row must not be consumed until newline completion.
        p2 = Path(td) / "partial.jsonl"
        first = json.dumps({"i": 1}).encode("utf-8") + b"\n"
        partial = json.dumps({"i": 2}).encode("utf-8")
        p2.write_bytes(first + partial)
        t2 = V15.BoundedJsonlTail(p2, max_rows=10, max_bytes=4096)
        a = t2.read_new()
        offset_after_partial = t2.offset
        with p2.open("ab") as fh:
            fh.write(b"\n")
        b = t2.read_new()
        checks["partial_line_waits_for_newline"] = (
            [x["i"] for x in a] == [1]
            and [x["i"] for x in b] == [2]
            and offset_after_partial == len(first)
        )

        stats = tail.stats()
        checks["stats_expose_backlog_and_limits"] = (
            stats["max_rows_per_read"] == 37
            and stats["max_bytes_per_read"] == 4096
            and "backlog_bytes" in stats
        )

    static = V15.static_self_check(show=False)
    checks["v15_static_ok"] = static.get("ok") is True
    checks["bounded_ingestion_declared"] = static.get("bounded_raw_ingestion") is True
    checks["alpha_unchanged_declared"] = static.get("alpha_rules_unchanged_from_v1_4") is True

    out = {
        **checks,
        "api_called": False,
        "orders_sent": False,
    }
    out["ok"] = all(bool(v) for k, v in checks.items())

    if show:
        for k, v in out.items():
            print(k, v)
    if not out["ok"]:
        raise RuntimeError(f"V1.5 bounded-tail unit failed: {out}")
    return out


if __name__ == "__main__":
    run(show=True)

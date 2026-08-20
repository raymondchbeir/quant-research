from __future__ import annotations

"""V1.5 bounded-ingestion / memory-telemetry hardening for deep-tail live trading.

This version changes no alpha or execution rule.  It keeps the V1.4 persistent-M5
state machine, V1.3 passive-GTC compatibility, 5c entries, first-fill-wins rule,
fixed JOIN_ASK/no-reprice exit, M1/M5 boundaries, quantity ladder, and loss logic.

The change is operational only:
- replace the unbounded ``JsonlTail.read_new()`` used by the strategy loop with a
  line-streaming reader that has strict per-call row and byte budgets;
- never ``read()`` the entire unread raw file and never split/materialize the whole
  backlog into Python objects at once;
- expose exact unread-byte backlog and cumulative rows/bytes consumed;
- publish process RSS and peak observed RSS into health.json once per second.

A separate V2.7 parent guardian enforces the runtime deadline and RSS ceiling even
if the main strategy loop is delayed.  This module itself does not place orders on
import.
"""

import json
import os
import subprocess
import time
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_4 as V14


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_5_BOUNDED_RAW_MEMORY_TELEMETRY"

BOOK_TAIL_MAX_ROWS_PER_READ = 2000
BOOK_TAIL_MAX_BYTES_PER_READ = 4 * 1024 * 1024
META_TAIL_MAX_ROWS_PER_READ = 500
META_TAIL_MAX_BYTES_PER_READ = 1 * 1024 * 1024
TELEMETRY_INTERVAL_S = 1.0


class BoundedJsonlTail:
    """Incremental JSONL tailer with bounded transient memory.

    Unlike the legacy tailer, this reader never calls ``fh.read()`` to EOF and
    never builds a list of every unread line.  A complete JSON line is consumed
    exactly once.  An incomplete writer-tail line is left unread until a newline
    appears on a later call.
    """

    def __init__(self, path, *, max_rows, max_bytes):
        self.path = Path(path)
        self.offset = 0
        self.max_rows = int(max_rows)
        self.max_bytes = int(max_bytes)
        if self.max_rows <= 0 or self.max_bytes <= 0:
            raise ValueError("max_rows and max_bytes must be positive")

        self.read_calls = 0
        self.rows_read = 0
        self.bytes_read = 0
        self.decode_errors = 0
        self.max_rows_returned = 0
        self.max_bytes_returned = 0

    def read_new(self):
        self.read_calls += 1
        if not self.path.exists():
            return []

        out = []
        bytes_this_call = 0
        last_complete_offset = int(self.offset)

        with self.path.open("rb") as fh:
            fh.seek(self.offset)

            while len(out) < self.max_rows:
                line_start = fh.tell()
                line = fh.readline()
                if not line:
                    break

                # The recorder may currently be writing the final JSON object.
                # Do not advance offset until the complete newline-terminated row
                # exists; retry this same row on the next call.
                if not line.endswith(b"\n"):
                    fh.seek(line_start)
                    break

                line_n = len(line)
                if bytes_this_call > 0 and bytes_this_call + line_n > self.max_bytes:
                    fh.seek(line_start)
                    break

                bytes_this_call += line_n
                last_complete_offset = fh.tell()

                raw = line[:-1]
                if not raw:
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except Exception:
                    self.decode_errors += 1
                    continue
                out.append(row)

                # One abnormally long but complete line may exceed max_bytes.  We
                # still consume that single row so the tail cannot deadlock on it.
                if bytes_this_call >= self.max_bytes:
                    break

        self.offset = int(last_complete_offset)
        self.rows_read += len(out)
        self.bytes_read += int(bytes_this_call)
        self.max_rows_returned = max(self.max_rows_returned, len(out))
        self.max_bytes_returned = max(self.max_bytes_returned, int(bytes_this_call))
        return out

    def backlog_bytes(self):
        try:
            size = int(self.path.stat().st_size)
        except Exception:
            return 0
        return max(0, size - int(self.offset))

    def stats(self):
        return {
            "path": str(self.path),
            "offset": int(self.offset),
            "backlog_bytes": int(self.backlog_bytes()),
            "max_rows_per_read": int(self.max_rows),
            "max_bytes_per_read": int(self.max_bytes),
            "read_calls": int(self.read_calls),
            "rows_read": int(self.rows_read),
            "bytes_read": int(self.bytes_read),
            "decode_errors": int(self.decode_errors),
            "max_rows_returned": int(self.max_rows_returned),
            "max_bytes_returned": int(self.max_bytes_returned),
        }


def _rss_bytes(pid=None):
    """Current resident-set size using POSIX ps; returns None if unavailable."""
    pid = int(pid or os.getpid())
    try:
        p = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if p.returncode != 0:
            return None
        text = p.stdout.strip().splitlines()
        if not text:
            return None
        # macOS/Linux ps report RSS in KiB.
        kib = float(text[0].strip().split()[0])
        if kib < 0:
            return None
        return int(kib * 1024.0)
    except Exception:
        return None


class BoundedMemoryEngine(V14.PersistentM5CleanupEngine):
    """V1.4 execution engine with bounded strategy-side raw ingestion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        raw = self.session / "raw_capture"
        # Parent construction has not consumed trading rows yet.  Replace the two
        # legacy unbounded tailers before the run loop begins.
        self.book_tail = BoundedJsonlTail(
            raw / "book_top3_events.jsonl",
            max_rows=BOOK_TAIL_MAX_ROWS_PER_READ,
            max_bytes=BOOK_TAIL_MAX_BYTES_PER_READ,
        )
        self.meta_tail = BoundedJsonlTail(
            raw / "market_metadata.jsonl",
            max_rows=META_TAIL_MAX_ROWS_PER_READ,
            max_bytes=META_TAIL_MAX_BYTES_PER_READ,
        )

        self._telemetry_next_wall = 0.0
        self._rss_peak_observed_bytes = 0
        self._lat(
            "BOUNDED_RAW_TAILS_INSTALLED",
            book_max_rows=BOOK_TAIL_MAX_ROWS_PER_READ,
            book_max_bytes=BOOK_TAIL_MAX_BYTES_PER_READ,
            meta_max_rows=META_TAIL_MAX_ROWS_PER_READ,
            meta_max_bytes=META_TAIL_MAX_BYTES_PER_READ,
        )

    def health(self, force=False):
        super().health(force=force)

        now = time.time()
        if not force and now < self._telemetry_next_wall:
            return
        self._telemetry_next_wall = now + TELEMETRY_INTERVAL_S

        rss = _rss_bytes()
        if rss is not None:
            self._rss_peak_observed_bytes = max(
                int(self._rss_peak_observed_bytes), int(rss)
            )

        try:
            h = B._read(self.health_path, {}) or {}
            h.update({
                "bounded_raw_ingestion": True,
                "strategy_rss_bytes": rss,
                "strategy_rss_mb": (rss / (1024.0 ** 2) if rss is not None else None),
                "strategy_rss_peak_observed_bytes": int(self._rss_peak_observed_bytes),
                "strategy_rss_peak_observed_mb": self._rss_peak_observed_bytes / (1024.0 ** 2),
                "book_tail": self.book_tail.stats(),
                "meta_tail": self.meta_tail.stats(),
            })
            B._atomic(self.health_path, h)
        except Exception:
            pass


def _install_patch():
    # Preserve every prior strategy/execution safety fix, replacing only the
    # strategy-side raw tail readers / telemetry engine class.
    V14._install_patch()
    V1.DeepTailLiveEngine = BoundedMemoryEngine


def run_live_process(session, cfg):
    _install_patch()
    old = V1.LIVE_VERSION
    try:
        V1.LIVE_VERSION = LIVE_VERSION
        return V1.run_live_process(Path(session).resolve(), cfg)
    finally:
        V1.LIVE_VERSION = old


def static_self_check(*, show=True):
    _install_patch()
    parent = V14.static_self_check(show=False)
    out = dict(parent)
    out.update({
        "version": LIVE_VERSION,
        "bounded_raw_ingestion": True,
        "legacy_read_to_eof_removed_from_strategy_tail": True,
        "book_tail_max_rows_per_read": BOOK_TAIL_MAX_ROWS_PER_READ,
        "book_tail_max_bytes_per_read": BOOK_TAIL_MAX_BYTES_PER_READ,
        "meta_tail_max_rows_per_read": META_TAIL_MAX_ROWS_PER_READ,
        "meta_tail_max_bytes_per_read": META_TAIL_MAX_BYTES_PER_READ,
        "rss_health_telemetry": True,
        "alpha_rules_unchanged_from_v1_4": True,
        "ok": bool(parent.get("ok")),
        "orders_sent": False,
    })
    if show:
        print("=" * 100)
        print("DEEP-TAIL LIVE V1.5 STATIC SELF-CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:62s}: {v}")
    return out


__all__ = [
    "LIVE_VERSION",
    "BOOK_TAIL_MAX_ROWS_PER_READ",
    "BOOK_TAIL_MAX_BYTES_PER_READ",
    "META_TAIL_MAX_ROWS_PER_READ",
    "META_TAIL_MAX_BYTES_PER_READ",
    "BoundedJsonlTail",
    "BoundedMemoryEngine",
    "run_live_process",
    "static_self_check",
]

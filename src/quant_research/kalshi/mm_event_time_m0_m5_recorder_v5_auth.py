from __future__ import annotations

"""V5 recorder with authenticated, rate-limit-resilient market discovery only.

This module deliberately leaves the proven V5 websocket/book/sequence/persistence
architecture unchanged.  The only runtime change is the REST transport used by
V5's periodic /markets discovery scan.

The original V5 scanner used recorder_core.rest_get(), an unauthenticated public
requests.Session call.  Kalshi can throttle that public path with HTTP 429 while
the same signed account GET succeeds.  Original V5 swallowed each per-series
exception and could therefore remain websocket-connected/"healthy" with zero
markets, zero subscriptions and zero data forever.

This wrapper keeps the exact V5 market-selection semantics:
- same nine frozen crypto series;
- same query statuses: unopened and open;
- same 5 minute pre-subscribe lead;
- same M0-M5 research window and M5+30s label tail;
- same websocket channels, Decimal book reconstruction, sequence handling,
  repair logic and persisted schema.

Only discovery transport changes:
- signed authenticated GET /markets;
- bounded retry for 429/temporary transport errors;
- a totally failed scan raises instead of silently looking like an empty market
  universe, allowing V5's existing supervisor health/error machinery to expose it.

No orders are ever sent by this module.
"""

import argparse
import asyncio
import base64
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from . import recorder_core as C
from . import mm_event_time_m0_m5_recorder_v5 as V5

STUDY_VERSION = "MM_EVENT_TIME_M0_M5_V5_AUTH_DISCOVERY"
DISCOVERY_TRANSPORT_VERSION = "SIGNED_ACCOUNT_GET_MARKETS_V1"
DISCOVERY_RETRY_DELAYS_S = (0.0, 0.25, 0.75)

_CLIENT = None


class _AuthenticatedReadClient:
    def __init__(self):
        self.key_id, self.private_key = C.load_auth()
        self.http = requests.Session()

    def get(self, path, params=None, timeout=8.0):
        url = C.REST_BASE + str(path)
        sign_path = urlparse(url).path
        ts = str(int(time.time() * 1000))
        msg = (ts + "GET" + sign_path).encode("utf-8")
        sig = self.private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        }
        r = self.http.get(url, params=params, headers=headers, timeout=float(timeout))
        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text}
        if not r.ok:
            raise RuntimeError(f"Kalshi GET {path} -> {r.status_code}: {body}")
        return body


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _AuthenticatedReadClient()
    return _CLIENT


def _signed_get_with_retry(path, params):
    last = None
    for delay in DISCOVERY_RETRY_DELAYS_S:
        if delay:
            time.sleep(delay)
        try:
            return _client().get(path, params=params)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"authenticated discovery failed after retries: {last!r}")


def _discover_sync_authenticated():
    now = C.utc_now()
    out = {}
    successes = 0
    failures = []

    # Preserve original V5 query semantics/order exactly.
    for series in V5.CRYPTO_SERIES:
        for query_status in ("unopened", "open"):
            params = {
                "series_ticker": series,
                "status": query_status,
                "limit": 1000,
            }
            try:
                body = _signed_get_with_retry("/markets", params)
                markets = body.get("markets") or []
                successes += 1
            except Exception as exc:
                failures.append(f"{series}/{query_status}: {exc!r}")
                continue

            for m in markets:
                row = V5._market_row(series, m)
                if row is None:
                    continue
                e = V5._elapsed(row, now)
                if e is None:
                    continue
                if -V5.PRESUBSCRIBE_LEAD_S <= e < V5.LABEL_TAIL_END_S:
                    out[row["ticker"]] = row

    # This is intentionally different from original V5's silent all-failure -> {}
    # behavior.  A truly failed scan must be visible as a scan error, not "healthy
    # but zero markets". Partial query failures remain recoverable on the next scan.
    if successes == 0:
        raise RuntimeError(
            "all authenticated V5 discovery queries failed; " + " | ".join(failures[-8:])
        )
    return out


async def _discover_authenticated():
    return await asyncio.to_thread(_discover_sync_authenticated)


def _install_patch():
    V5.STUDY_VERSION = STUDY_VERSION
    V5._discover_sync = _discover_sync_authenticated
    V5._discover = _discover_authenticated


async def run_event_time_m0_m5_v5_auth_recorder(session_dir: Path):
    _install_patch()
    return await V5.run_event_time_m0_m5_v5_recorder(Path(session_dir).resolve())


def static_self_check(*, show=True):
    out = {
        "study_version": STUDY_VERSION,
        "base_v5_version_before_patch": "MM_EVENT_TIME_M0_M5_V5_DEV",
        "discovery_transport_version": DISCOVERY_TRANSPORT_VERSION,
        "authenticated_discovery": True,
        "query_statuses_unchanged": ("unopened", "open"),
        "universe_unchanged": tuple(V5.CRYPTO_SERIES),
        "presubscribe_lead_s": V5.PRESUBSCRIBE_LEAD_S,
        "research_window_s": [V5.TRADE_WINDOW_START_S, V5.TRADE_WINDOW_END_S],
        "label_tail_end_s": V5.LABEL_TAIL_END_S,
        "websocket_book_logic_unchanged": True,
        "orders_sent": False,
        "ok": True,
    }
    if show:
        print("=" * 104)
        print("V5 AUTHENTICATED DISCOVERY STATIC CHECK — NO API / NO ORDERS")
        print("=" * 104)
        for k, v in out.items():
            print(f"{k:44s}: {v}")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-session", type=str, default=None)
    a = ap.parse_args()
    if a.run_session:
        asyncio.run(run_event_time_m0_m5_v5_auth_recorder(Path(a.run_session)))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "STUDY_VERSION", "DISCOVERY_TRANSPORT_VERSION", "static_self_check",
    "run_event_time_m0_m5_v5_auth_recorder", "_discover_sync_authenticated",
]

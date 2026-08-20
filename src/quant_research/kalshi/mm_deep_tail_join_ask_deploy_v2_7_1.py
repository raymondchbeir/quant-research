from __future__ import annotations

"""V2.7.1 parent-side preflight hardening for transient public API 429s.

No live strategy, execution, sizing, timing, M5, memory, guardian, or promotion
mechanic changes from V2.7/V1.5.  This wrapper only hardens the read-only fee
preflight by pacing requests, retrying 429s, and briefly caching a successful
snapshot so notebook preflight + immediate launch do not duplicate the public GET
burst.  Importing this module sends no orders.
"""

import contextlib
import time

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_deploy_v2_7 as V27


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_7_1_RESILIENT_FEE_PREFLIGHT"
LIVE = V27.LIVE
CORE = V27.CORE
Q1_ARM = V27.Q1_ARM
Q10_ARM = V27.Q10_ARM
KILL_ARM = V27.KILL_ARM
PROMOTION_PATH = V27.PROMOTION_PATH

# Capture the unwrapped V2.7 function once.  start_q1_smoke/start_q10 temporarily
# replace V27.live_preflight; calling V27.live_preflight from our wrapper would
# recurse.  Always delegate to this saved original instead.
_V27_LIVE_PREFLIGHT_ORIGINAL = V27.live_preflight

FEE_CACHE_TTL_S = 120.0
FEE_REQUEST_PACE_S = 0.40
FEE_RETRY_DELAYS_S = (0.75, 1.5, 3.0, 6.0, 10.0)
_FEE_CACHE = {"time": 0.0, "result": None}


def _is_429(exc):
    text = repr(exc).lower()
    return "429" in text or "too many requests" in text


def _rest_get_429_resilient(path, params=None):
    params = params or {}
    last = None
    attempts = len(FEE_RETRY_DELAYS_S) + 1
    for i in range(attempts):
        try:
            out = C.rest_get(path, params)
            time.sleep(FEE_REQUEST_PACE_S)
            return out
        except Exception as exc:
            last = exc
            if not _is_429(exc):
                raise
            if i >= len(FEE_RETRY_DELAYS_S):
                break
            delay = float(FEE_RETRY_DELAYS_S[i])
            print(
                f"Fee preflight HTTP 429 on {path}; "
                f"retrying in {delay:.2f}s ({i + 1}/{len(FEE_RETRY_DELAYS_S)})..."
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Fee preflight exhausted HTTP-429 retries for {path}: {last!r}"
    )


def resilient_fee_preflight(*, horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
                            save_path=None, show=True):
    now_wall = time.time()
    cached = _FEE_CACHE.get("result")
    age = now_wall - float(_FEE_CACHE.get("time") or 0.0)

    if cached is not None and age <= FEE_CACHE_TTL_S:
        out = dict(cached)
        out["cache_hit"] = True
        out["cache_age_s"] = age
        if save_path is not None:
            OOS._atomic_json(save_path, out)
        if show:
            print(
                f"FEE PREFLIGHT: PASS (cached {age:.1f}s old; "
                "no repeated public API burst)"
            )
        return out

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    problems = []
    multipliers = {}

    for series in OOS.SERIES:
        try:
            payload = _rest_get_429_resilient(f"/series/{series}", {})
            s = payload.get("series") or {}
        except Exception as exc:
            problems.append(f"{series}: get-series failed: {exc!r}")
            continue

        fee_type = str(s.get("fee_type") or "").strip().lower()
        try:
            mult = float(s.get("fee_multiplier"))
        except Exception:
            mult = np.nan

        if fee_type != "quadratic":
            problems.append(
                f"{series}: fee_type={fee_type!r}, expected 'quadratic' "
                "(zero-maker structure)"
            )
        if not np.isfinite(mult) or mult <= 0:
            problems.append(
                f"{series}: invalid fee_multiplier={s.get('fee_multiplier')!r}"
            )
        else:
            multipliers[series] = float(mult)

        upcoming = []
        try:
            fc = _rest_get_429_resilient(
                "/series/fee_changes",
                {"series_ticker": series, "show_historical": False},
            )
            upcoming = fc.get("series_fee_change_arr") or []
        except Exception as exc:
            problems.append(
                f"{series}: fee-change preflight failed: {exc!r}"
            )

        near = []
        for r in upcoming:
            t = pd.to_datetime(r.get("scheduled_ts"), utc=True, errors="coerce")
            if pd.isna(t):
                continue
            hours = (t - now).total_seconds() / 3600.0
            if -1e-9 <= hours <= float(horizon_hours):
                near.append(r)
        if near:
            problems.append(
                f"{series}: scheduled fee change within "
                f"{float(horizon_hours):.0f}h: {near}"
            )

        rows.append({
            "series": series,
            "fee_type": fee_type,
            "fee_multiplier": mult,
            "last_updated_ts": s.get("last_updated_ts"),
            "upcoming_fee_changes": upcoming,
            "near_horizon_fee_changes": near,
        })

    ok = len(problems) == 0 and len(multipliers) == len(OOS.SERIES)
    out = {
        "time": OOS._iso_ts(),
        "ok": ok,
        "horizon_hours": float(horizon_hours),
        "series": rows,
        "multipliers": multipliers,
        "problems": problems,
        "guardrail": (
            "Startup refuses unless maker-fee structure is verified for every "
            "frozen series; transient 429s are retried, never ignored."
        ),
        "resilient_429_retry": True,
        "cache_hit": False,
        "cache_age_s": 0.0,
    }

    if save_path is not None:
        OOS._atomic_json(save_path, out)
    if show:
        print("FEE PREFLIGHT:", "PASS" if ok else "FAIL")
        for r in rows:
            print(
                f"  {r['series']:10s} "
                f"type={r['fee_type']!s:28s} "
                f"multiplier={r['fee_multiplier']}"
            )
        for p in problems:
            print("  ERROR:", p)
    if not ok:
        raise RuntimeError(
            "Fee preflight failed; refusing live startup. " + " | ".join(problems)
        )

    _FEE_CACHE["time"] = time.time()
    _FEE_CACHE["result"] = dict(out)
    return out


@contextlib.contextmanager
def _patched_fee_preflight():
    old = OOS.fee_preflight
    OOS.fee_preflight = resilient_fee_preflight
    try:
        yield
    finally:
        OOS.fee_preflight = old


def static_self_check(*, show=True):
    base = V27.static_self_check(show=False)
    out = dict(base)
    out.update({
        "deploy_wrapper_version": DEPLOY_VERSION,
        "resilient_fee_preflight": True,
        "fee_429_retry_delays_s": FEE_RETRY_DELAYS_S,
        "fee_request_pace_s": FEE_REQUEST_PACE_S,
        "fee_cache_ttl_s": FEE_CACHE_TTL_S,
        "fee_429_never_bypassed": True,
        "v27_delegate_recursion_safe": _V27_LIVE_PREFLIGHT_ORIGINAL is not live_preflight,
        "orders_sent": False,
        "ok": bool(base.get("ok")),
    })
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.7.1 STATIC CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:50s}: {v}")
    if not out["ok"] or out["v27_delegate_recursion_safe"] is not True:
        raise RuntimeError(f"V2.7.1 static self-check failed: {out}")
    return out


def live_preflight(**kwargs):
    with _patched_fee_preflight():
        out = _V27_LIVE_PREFLIGHT_ORIGINAL(**kwargs)
    out = dict(out)
    out["parent_preflight_wrapper"] = DEPLOY_VERSION
    return out


def _with_resilient_v27_launch(fn, **kwargs):
    old = V27.live_preflight
    V27.live_preflight = live_preflight
    try:
        return fn(**kwargs)
    finally:
        V27.live_preflight = old


def start_q1_smoke(**kwargs):
    return _with_resilient_v27_launch(V27.start_q1_smoke, **kwargs)


def start_q10_one_hour(**kwargs):
    return _with_resilient_v27_launch(V27.start_q10_one_hour, **kwargs)


def q1_promotion_check(*args, **kwargs):
    return V27.q1_promotion_check(*args, **kwargs)


def live_status(**kwargs):
    return V27.live_status(**kwargs)


def kill_and_flatten_live(**kwargs):
    return V27.kill_and_flatten_live(**kwargs)


def api_capacity_preflight(**kwargs):
    return V27.api_capacity_preflight(**kwargs)


__all__ = [
    "DEPLOY_VERSION", "LIVE", "CORE", "Q1_ARM", "Q10_ARM", "KILL_ARM",
    "PROMOTION_PATH", "resilient_fee_preflight", "static_self_check",
    "live_preflight", "api_capacity_preflight", "start_q1_smoke",
    "start_q10_one_hour", "q1_promotion_check", "live_status",
    "kill_and_flatten_live",
]

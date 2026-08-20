from __future__ import annotations

"""Operational overnight wrapper: frozen M1->M5 Q10 live strategy + fresh M0->M15 sidecar.

This module changes NO alpha/execution mechanics in the live engine.  It is a thin
orchestration layer around the already-tested V1.5/V2.7.1 stack.

Overnight plan
--------------
- Require the existing V1.5 Q1 operational-promotion receipt at the exact current git HEAD.
- Run the frozen M1->M5 deep-tail strategy at Q10 for exactly 8 trading hours.
- Keep the V1.5 external guardian active: 1 GiB RSS warning, 2 GiB hard RSS intervention,
  and deadline+5s independent watchdog.
- Use a $20 SOFTWARE equity-loss trigger.  This is not a guaranteed final-loss cap because
  exchange movement, cleanup latency, and slippage can exceed the trigger.
- Simultaneously launch the existing corrected M0->M15 exploratory recorder so the morning
  has fresh full-window data for the separate M5->M12 research candidate.
- Launch an independent watcher process.  When the Q10 live process exits, the watcher stops
  and preserves the full-window recorder session automatically.

Scientific separation
---------------------
The live strategy remains the frozen M1->M5 candidate.  The M0->M15 sidecar is data
collection only and must not change the overnight strategy.  Fresh sidecar data may later
be used for forward/shadow evaluation of the separate M5->M12 candidate.

Importing this module sends no orders and starts no recorder.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from quant_research.kalshi import recorder_core as C
from quant_research.kalshi import mm_cycle_q10_live_strategy_v1 as B
from quant_research.kalshi import mm_cycle_q10_live_strategy_v10 as V10
from quant_research.kalshi import mm_deep_tail_join_ask_deploy_v2_7 as V27
from quant_research.kalshi import mm_deep_tail_join_ask_deploy_v2_7_1 as V271
from quant_research.kalshi import mm_event_time_m0_m15_exploratory_recorder_v1 as FULL15

VERSION = "MM_DEEP_TAIL_Q10_8H_OVERNIGHT_V1"
EXPECTED_BASE_HEAD = "3ed16f61e183ad41b32e8c5beeedff9797f7cf2f"
ARM = "LIVE_DEEP_TAIL_Q10_8H_OVERNIGHT_V15"
KILL_ARM = V27.KILL_ARM
RUNTIME_HOURS = 8.0
QUOTE_SIZE = 10.0
MAX_START_LOSS_USD = 20.0
MIN_START_EQUITY_USD = 75.0
MIN_FREE_DISK_GB = 15.0
WATCH_POLL_S = 2.0

def _atomic(path: Path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)

def _read(path: Path, default=None):
    try: return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception: return default

def _current_head():
    return (V10._git_state() or {}).get("head")

def _free_disk_gb():
    return float(shutil.disk_usage(C.DATA_ROOT).free) / (1024.0 ** 3)

def static_self_check(*, show=True):
    base = V271.static_self_check(show=False)
    head = _current_head()
    checks = {
        "version": VERSION,
        "base_static_ok": bool(base.get("ok")),
        "expected_base_head": EXPECTED_BASE_HEAD,
        "current_head": head,
        "exact_base_head": bool(head and str(head) == EXPECTED_BASE_HEAD),
        "live_engine_version": V27.LIVE.LIVE_VERSION,
        "q10": QUOTE_SIZE == 10.0,
        "runtime_hours": RUNTIME_HOURS == 8.0,
        "software_loss_trigger_usd": MAX_START_LOSS_USD == 20.0,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "guardian_separate_process": base.get("guardian_separate_process") is True,
        "guardian_rss_hard_limit_mb": V27.RSS_HARD_LIMIT_MB,
        "guardian_deadline_grace_s": V27.GUARDIAN_DEADLINE_GRACE_S,
        "resilient_fee_preflight": base.get("resilient_fee_preflight") is True,
        "full15_study_version": FULL15.STUDY_VERSION,
        "full15_capture_end_s": FULL15.TRADE_WINDOW_END_S,
        "full15_is_data_only": True,
        "orders_sent": False,
    }
    ok = (checks["base_static_ok"] and checks["exact_base_head"] and checks["q10"] and checks["runtime_hours"] and checks["software_loss_trigger_usd"] and checks["bounded_raw_ingestion"] and checks["guardian_separate_process"] and checks["resilient_fee_preflight"] and float(checks["full15_capture_end_s"]) == 900.0)
    out = {**checks, "ok": bool(ok)}
    if show:
        print("="*100); print("Q10 8H OVERNIGHT STATIC CHECK — NO ORDERS"); print("="*100)
        for k,v in out.items(): print(f"{k:40s}: {v}")
    if not ok: raise RuntimeError(f"Overnight static self-check failed: {out}")
    return out

def _require_q1_promotion():
    return V27._require_q1_promotion()

def _launch_q10_8h(*, arm_phrase, min_start_equity_usd):
    if str(arm_phrase) != ARM: raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={ARM!r} exactly.")
    if abs(float(min_start_equity_usd)-MIN_START_EQUITY_USD)>1e-12: raise RuntimeError(f"Overnight V1 is frozen to min_start_equity_usd={MIN_START_EQUITY_USD:.2f}.")
    _require_q1_promotion()
    old = V27.live_preflight
    V27.live_preflight = V271.live_preflight
    try:
        return V27._launch(q=QUOTE_SIZE, hours=RUNTIME_HOURS, max_loss=MAX_START_LOSS_USD, min_equity=MIN_START_EQUITY_USD, mode="DEEP_TAIL_Q10_8H_OVERNIGHT_V15", arm_phrase=arm_phrase, expected_arm=ARM)
    finally:
        V27.live_preflight = old

def _watcher_log_path(live_session: Path): return Path(live_session)/"overnight_full15_watcher_v1.log"
def _watcher_receipt_path(live_session: Path): return Path(live_session)/"overnight_full15_watcher_v1.json"

def _launch_watcher(live_session: Path, live_pid: int, full15_session: Path):
    script = Path(__file__).resolve(); log = _watcher_log_path(live_session)
    env = dict(os.environ); src = str((C.PROJECT_ROOT/"src").resolve()); old_pp = env.get("PYTHONPATH","")
    env["PYTHONPATH"] = src if not old_pp else src + os.pathsep + old_pp
    cmd = [sys.executable, str(script), "--watch-live-pid", str(int(live_pid)), "--live-session", str(Path(live_session).resolve()), "--full15-session", str(Path(full15_session).resolve())]
    fh = log.open("a", buffering=1, encoding="utf-8")
    try: p = subprocess.Popen(cmd, cwd=str(C.PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, env={**env,"PYTHONUNBUFFERED":"1"})
    finally: fh.close()
    return p,cmd,log

def _watch_loop(*, live_pid:int, live_session:Path, full15_session:Path):
    live_pid=int(live_pid); live_session=Path(live_session).resolve(); full15_session=Path(full15_session).resolve(); started=time.time()
    while B._pid_alive(live_pid): time.sleep(WATCH_POLL_S)
    stop_error=None; stopped_session=None
    try: stopped_session=FULL15.stop_full15_recording(expected_session=full15_session, timeout_s=FULL15.STOP_TIMEOUT_S)
    except Exception as exc: stop_error=repr(exc)
    receipt={"time":B._iso(),"version":VERSION,"live_pid":live_pid,"live_session":str(live_session),"full15_session":str(full15_session),"live_process_dead":not B._pid_alive(live_pid),"full15_stop_return":str(stopped_session) if stopped_session else None,"full15_stop_error":stop_error,"watch_runtime_s":time.time()-started}
    _atomic(_watcher_receipt_path(live_session),receipt); print(json.dumps(receipt,indent=2),flush=True); return receipt

def start_overnight(*, arm_phrase=None, min_start_equity_usd=MIN_START_EQUITY_USD):
    static_self_check(show=True); promotion=_require_q1_promotion()
    live_now=V271.live_status(show=False)
    if live_now.get("running"): raise RuntimeError(f"Another deep-tail live process is already running: {live_now.get('control')}")
    full_now=FULL15.full15_status(show=False)
    if full_now.get("running"): raise RuntimeError("A Full15 exploratory recorder is already running. Stop or preserve it explicitly before overnight start.")
    free_gb=_free_disk_gb()
    if free_gb<MIN_FREE_DISK_GB: raise RuntimeError(f"Overnight start refused: only {free_gb:.1f} GiB free under DATA_ROOT; requires >= {MIN_FREE_DISK_GB:.1f} GiB.")
    print("\n"+"="*100); print("STARTING FRESH FULL-WINDOW DATA SIDECAR — NO ORDERS"); print("="*100)
    full=FULL15.start_full15_recording(); full_session=Path(full["session_dir"]).resolve()
    try:
        print("\n"+"="*100); print("STARTING REAL-MONEY Q10 8H OVERNIGHT"); print("="*100)
        live=_launch_q10_8h(arm_phrase=arm_phrase,min_start_equity_usd=min_start_equity_usd)
    except Exception:
        try: FULL15.stop_full15_recording(expected_session=full_session)
        except Exception as cleanup_exc: print("WARNING: Q10 launch failed and Full15 cleanup also failed:",repr(cleanup_exc))
        raise
    ctl=live.get("control") or {}; live_session=Path(ctl["session_dir"]).resolve(); live_pid=int(ctl["pid"])
    watcher,watcher_cmd,watcher_log=_launch_watcher(live_session,live_pid,full_session)
    receipt={"time":B._iso(),"version":VERSION,"base_head":_current_head(),"q1_promotion_receipt":promotion,"live_session":str(live_session),"live_pid":live_pid,"live_mode":ctl.get("mode"),"quote_size":QUOTE_SIZE,"runtime_hours":RUNTIME_HOURS,"software_loss_trigger_usd":MAX_START_LOSS_USD,"software_loss_trigger_not_guaranteed_final_cap":True,"full15_session":str(full_session),"full15_pid":full.get("pid"),"full15_capture":"M0 <= elapsed < M15, data only","watcher_pid":watcher.pid,"watcher_log":str(watcher_log),"watcher_command":watcher_cmd,"free_disk_gb_at_start":free_gb}
    _atomic(live_session/"overnight_orchestrator_v1.json",receipt)
    print("\n"+"="*100); print("OVERNIGHT STACK ARMED"); print("="*100)
    print("Live strategy:             frozen M1->M5 deep-tail JOIN_ASK"); print("Live size:                 Q10"); print("Trading runtime:           8.0h from first complete window"); print("Software loss trigger:     -$20.00 (NOT a guaranteed final-loss cap)"); print("Live session:              ",live_session); print("Live PID:                  ",live_pid); print("Guardian:                  V1.5/V2.7 RSS + deadline guardian ACTIVE"); print("Fresh research recorder:   M0->M15 full-window sidecar"); print("Full15 session:            ",full_session); print("Full15 PID:                ",full.get("pid")); print("Auto-stop watcher PID:     ",watcher.pid); print("Auto-scale:                OFF"); print("M5->M12 strategy trading:  NO — DATA COLLECTION ONLY"); print("="*100)
    return overnight_status(show=False)

def overnight_status(*, show=True, tail_lines=40):
    live=V271.live_status(show=False,tail_lines=tail_lines); full=FULL15.full15_status(show=False); ctl=live.get("control") or {}; live_session=Path(ctl.get("session_dir","")) if ctl.get("session_dir") else None
    orch=_read(live_session/"overnight_orchestrator_v1.json",{}) if live_session else {}; watcher=_read(_watcher_receipt_path(live_session),{}) if live_session else {}
    out={"version":VERSION,"live":live,"full15":full,"orchestrator":orch or {},"watcher_receipt":watcher or {}}
    if show:
        h=live.get("health") or {}; guardian=live.get("guardian") or {}; final=live.get("final_summary") or {}
        print("="*100); print("Q10 8H OVERNIGHT STATUS"); print("="*100)
        print("LIVE RUNNING:              ",live.get("running")); print("Live session:              ",ctl.get("session_dir")); print("Mode:                      ",ctl.get("mode")); print("Q:                         ",(ctl.get("config") or {}).get("quote_size")); print("Configured hours:          ",(ctl.get("config") or {}).get("runtime_hours")); print("Trade deadline:            ",h.get("trade_deadline")); print("Equity:                    ",h.get("equity")); print("Peak drawdown:             ",h.get("max_peak_drawdown")); print("Positions:                 ",h.get("positions")); print("Active tracks:             ",len(h.get("active_tracks") or {})); print("Strategy RSS MB:           ",h.get("strategy_rss_mb")); print("Strategy RSS peak MB:      ",h.get("strategy_rss_peak_observed_mb")); print("Guardian total RSS MB:     ",((guardian.get("rss") or {}).get("total_rss_mb"))); print("Guardian peak RSS MB:      ",guardian.get("peak_total_rss_mb")); print("Guardian intervened:       ",guardian.get("intervened")); print("Last error:                ",h.get("last_error"))
        if final: print("Shutdown reason:           ",final.get("shutdown_reason")); print("Flat verified:             ",final.get("flat_verified"))
        print(); print("FULL15 RUNNING:            ",full.get("running")); print("Full15 session:            ",full.get("session_dir")); print("Full15 PID:                ",full.get("pid")); print("Full15 healthy:            ",full.get("healthy")); print("Full15 book rows:          ",((full.get("counts") or {}).get("book_rows"))); print("Full15 trade rows:         ",((full.get("counts") or {}).get("trade_rows"))); print("Full15 last error:         ",full.get("last_error")); print("Auto-stop watcher receipt: ",bool(watcher)); print("="*100)
    return out

def emergency_stop(*, arm_phrase=None):
    if str(arm_phrase)!=KILL_ARM: raise RuntimeError(f"Emergency stop refused. Pass arm_phrase={KILL_ARM!r} exactly.")
    live_receipt=None; live_error=None
    try: live_receipt=V271.kill_and_flatten_live(arm_phrase=KILL_ARM,wait_s=3.0,term_wait_s=6.0,kill_wait_s=3.0)
    except Exception as exc: live_error=repr(exc)
    full_error=None; full_stopped=None
    try:
        fs=FULL15.full15_status(show=False)
        if fs.get("running"): full_stopped=FULL15.stop_full15_recording(expected_session=fs.get("session_dir"))
    except Exception as exc: full_error=repr(exc)
    out={"time":B._iso(),"live_receipt":live_receipt,"live_error":live_error,"full15_stopped":str(full_stopped) if full_stopped else None,"full15_error":full_error}
    if live_error: raise RuntimeError(f"Live emergency stop error: {live_error}; full15_error={full_error}")
    return out

def morning_summary(*, show=True):
    out=overnight_status(show=show,tail_lines=80); live=out["live"]; full=out["full15"]
    if live.get("running"): raise RuntimeError("Overnight live process is still running; do not treat this as a completed run.")
    if full.get("running"):
        print("WARNING: live is stopped but Full15 sidecar is still running; stopping it now.")
        FULL15.stop_full15_recording(expected_session=full.get("session_dir")); out=overnight_status(show=show,tail_lines=80)
    return out

def _main():
    ap=argparse.ArgumentParser(); ap.add_argument("--watch-live-pid",type=int); ap.add_argument("--live-session",type=str); ap.add_argument("--full15-session",type=str); args=ap.parse_args()
    if args.watch_live_pid:
        if not args.live_session or not args.full15_session: raise SystemExit("watcher requires --live-session and --full15-session")
        _watch_loop(live_pid=args.watch_live_pid,live_session=Path(args.live_session),full15_session=Path(args.full15_session))

if __name__=="__main__": _main()

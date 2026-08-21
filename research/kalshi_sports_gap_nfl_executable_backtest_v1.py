from __future__ import annotations

"""Causal executable NFL underdog-reversion backtest on one-minute Kalshi BBO closes.

Execution-economics screen, not an alpha proof. Source has one-minute YES bid/ask
closes but no displayed size/depth or sub-minute quote path, so this is Q1 only.
Signals use completed BBO observations; entry and exit are delayed causally and
executed at underdog ask/bid. Current KXNFLGAME quadratic taker fees are applied.
No API calls. No orders. No live-Q50 imports.
"""

import argparse, json, math
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any
import numpy as np
import pandas as pd

VERSION = "KALSHI_SPORTS_NFL_EXECUTABLE_BACKTEST_V1_CAUSAL_Q1"
SERIES = "KXNFLGAME"
QTY = 1.0
TAKER_MULTIPLIER = 1.0
SPREAD_CAPS = (0.01, 0.02)
LATENCIES_MIN = (0, 1, 2)
TIME_BINS_MIN = (0, 30, 60, 90, 120, 150, 180, math.inf)
TIME_LABELS = ("0-30", "30-60", "60-90", "90-120", "120-150", "150-180", "180+")
FAV_BINS = (0.50, 0.55, 0.60, 0.70, 0.80, 0.90)
FAV_LABELS = ("50-55", "55-60", "60-70", "70-80", "80-90")
TARGETS_C = (5, 10, 15)
MAX_HOLDS_MIN = (5, 10, 20, 30)
MAX_QUOTE_LAG_S = 90.0
EPS = 1e-12


def _f(x: Any, default: float = np.nan) -> float:
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _atomic_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def quadratic_taker_fee(price: float, qty: float = QTY, multiplier: float = TAKER_MULTIPLIER) -> float:
    price, qty, multiplier = float(price), float(qty), float(multiplier)
    if qty <= 0 or not np.isfinite(price) or not 0 <= price <= 1:
        return np.nan
    raw = 0.07 * multiplier * qty * price * (1.0 - price)
    return math.ceil(max(0.0, raw) * 10000.0 - 1e-12) / 10000.0


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(1 - alpha / 2); p = k / n; den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return max(0.0, center-half), min(1.0, center+half)


def mean_ci(values: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(x): return np.nan, np.nan
    m = float(np.mean(x))
    if len(x) == 1: return m, m
    z = NormalDist().inv_cdf(1-alpha/2); se = float(np.std(x, ddof=1)/math.sqrt(len(x)))
    return m-z*se, m+z*se


def season_regime(ts: float) -> str:
    t = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
    if pd.isna(t): return "unknown"
    if int(t.month) in (7, 8): return "preseason"
    if int(t.month) in (9,10,11,12,1,2): return "regular_or_postseason"
    return "unknown"


def _fav_state(mid: float) -> tuple[str | None, float]:
    if not np.isfinite(mid): return None, np.nan
    if mid > 0.5 + EPS: return "YES", float(mid)
    if mid < 0.5 - EPS: return "NO", float(1-mid)
    return None, 0.5


def _band_for_prob(p: float) -> str | None:
    if not np.isfinite(p): return None
    for i, label in enumerate(FAV_LABELS):
        if FAV_BINS[i] <= p < FAV_BINS[i+1]: return label
    return None


def _bucket_for_elapsed_min(x: float) -> str | None:
    if not np.isfinite(x) or x < 0: return None
    for lo, hi, label in zip(TIME_BINS_MIN[:-1], TIME_BINS_MIN[1:], TIME_LABELS):
        if lo <= x < hi: return label
    return None


def _underdog_bid_ask(yes_bid: float, yes_ask: float, favorite_side: str) -> tuple[str,float,float]:
    if favorite_side == "YES": return "NO", float(1-yes_ask), float(1-yes_bid)
    if favorite_side == "NO": return "YES", float(yes_bid), float(yes_ask)
    raise ValueError(favorite_side)


def _underdog_bid(yes_bid: float, yes_ask: float, underdog_side: str) -> float:
    return float(yes_bid) if underdog_side == "YES" else float(1-yes_ask)


def static_self_check(show=True):
    side, b, a = _underdog_bid_ask(.69, .71, "YES")
    out = {
        "version": VERSION, "offline_only": True, "api_called": False, "orders_sent": False,
        "live_q50_modules_imported": False, "series_filter": SERIES, "qty_contracts": QTY,
        "top_of_book_only": True, "depth_available": False, "capacity_claim_above_q1": False,
        "signal_uses_completed_minute_bbo_close": True, "entry_crosses_executable_underdog_ask": True,
        "exit_crosses_executable_underdog_bid": True, "entry_revalidated_after_latency": True,
        "exit_latency_applied": True, "one_trade_per_game_per_exact_rule": True,
        "settlement_fallback_explicit": True,
        "fee_formula": "ceil_0.0001(0.07 * multiplier * qty * p * (1-p))",
        "taker_multiplier": TAKER_MULTIPLIER,
        "complement_check": side == "NO" and abs(b-.29)<1e-12 and abs(a-.31)<1e-12,
        "fee_50_check": abs(quadratic_taker_fee(.5)-.0175)<1e-12,
        "fee_20_check": abs(quadratic_taker_fee(.2)-.0112)<1e-12,
    }
    out["ok"] = bool(out["complement_check"] and out["fee_50_check"] and out["fee_20_check"])
    if show:
        print("="*132); print("NFL EXECUTABLE UNDERDOG-REVERSION BACKTEST — STATIC CHECK"); print("="*132)
        for k,v in out.items(): print(f"{k:54s}: {v}")
    if not out["ok"]: raise RuntimeError("Static self-check failed")
    return out


def load_inputs(root: Path):
    paths = pd.read_csv(root/"minute_paths.csv.gz", compression="gzip")
    markets = pd.read_csv(root/"markets.csv")
    paths = paths[paths.series_ticker.astype(str) == SERIES].copy()
    markets = markets[markets.series_ticker.astype(str) == SERIES].copy()
    for c in ("yes_bid","yes_ask","yes_mid","quote_spread","end_period_ts","elapsed_from_start_s","game_start_ts"):
        paths[c] = pd.to_numeric(paths[c], errors="coerce")
    paths = paths[(paths.phase.astype(str)=="in_game") & (paths.elapsed_from_start_s>=0)].copy()
    valid = (paths.yes_bid.notna() & paths.yes_ask.notna() & (paths.yes_bid>=0) &
             (paths.yes_ask<=1) & (paths.yes_bid<paths.yes_ask) &
             paths.yes_mid.between(0,1,inclusive="both") & (paths.quote_spread>=0))
    paths = paths[valid].sort_values(["ticker","end_period_ts"]).reset_index(drop=True)
    markets["settlement_value"] = pd.to_numeric(markets.get("settlement_value"), errors="coerce")
    markets["game_start_ts"] = pd.to_numeric(markets.get("game_start_ts"), errors="coerce")
    if paths.empty: raise RuntimeError("No valid NFL in-game BBO rows")
    return paths, markets


def build_market_arrays(paths, markets):
    meta = markets.drop_duplicates("ticker").set_index("ticker").to_dict("index")
    out = {}
    for ticker,g in paths.groupby("ticker", sort=False):
        ticker=str(ticker); g=g.sort_values("end_period_ts").reset_index(drop=True)
        starts=g.game_start_ts.dropna(); gs=float(starts.iloc[0]) if len(starts) else _f(meta.get(ticker,{}).get("game_start_ts"))
        out[ticker]={"ticker":ticker,"event_ticker":str(g.event_ticker.iloc[0]),
            "ts":g.end_period_ts.to_numpy(float),"bid":g.yes_bid.to_numpy(float),"ask":g.yes_ask.to_numpy(float),
            "mid":g.yes_mid.to_numpy(float),"spread":g.quote_spread.to_numpy(float),
            "elapsed_min":g.elapsed_from_start_s.to_numpy(float)/60.0,"game_start_ts":gs,
            "season_regime":season_regime(gs),"settlement_value":_f(meta.get(ticker,{}).get("settlement_value"))}
    return out


def _first_index_at_or_after(ts, target, max_lag=MAX_QUOTE_LAG_S):
    i=int(np.searchsorted(ts,target,side="left"))
    if i>=len(ts) or ts[i]-target>max_lag+EPS: return None
    return i


def first_signal_index(m, spread_cap, time_bucket, start_band):
    for i in range(len(m["ts"])):
        if m["spread"][i] > spread_cap+EPS: continue
        if _bucket_for_elapsed_min(m["elapsed_min"][i]) != time_bucket: continue
        side,p=_fav_state(m["mid"][i])
        if side is not None and _band_for_prob(p)==start_band: return i
    return None


def entry_attempt(m, spread_cap, time_bucket, start_band, latency_min):
    si=first_signal_index(m,spread_cap,time_bucket,start_band)
    if si is None: return None
    sside,sp=_fav_state(m["mid"][si]); sts=float(m["ts"][si]); desired=sts+latency_min*60.0
    row={"ticker":m["ticker"],"event_ticker":m["event_ticker"],"season_regime":m["season_regime"],
         "spread_cap":float(spread_cap),"time_bucket":time_bucket,"start_band":start_band,"latency_min":int(latency_min),
         "signal_index":int(si),"signal_ts":sts,"signal_elapsed_min":float(m["elapsed_min"][si]),
         "signal_yes_mid":float(m["mid"][si]),"signal_spread":float(m["spread"][si]),
         "signal_favorite_side":sside,"signal_favorite_prob":float(sp),"desired_entry_ts":desired,
         "entered":False,"entry_skip_reason":None}
    ei=_first_index_at_or_after(m["ts"],desired)
    if ei is None: row["entry_skip_reason"]="no_fresh_quote_after_latency"; return row
    eside,ep=_fav_state(m["mid"][ei])
    row.update({"entry_index":int(ei),"entry_ts":float(m["ts"][ei]),"entry_quote_lag_s":float(m["ts"][ei]-desired),
                "entry_elapsed_min":float(m["elapsed_min"][ei]),"entry_yes_mid":float(m["mid"][ei]),
                "entry_spread":float(m["spread"][ei]),"entry_favorite_side":eside,"entry_favorite_prob":float(ep)})
    if eside!=sside: row["entry_skip_reason"]="favorite_side_changed"; return row
    if _band_for_prob(ep)!=start_band: row["entry_skip_reason"]="favorite_left_signal_band"; return row
    if m["spread"][ei]>spread_cap+EPS: row["entry_skip_reason"]="spread_widened"; return row
    uside,ubid,uask=_underdog_bid_ask(float(m["bid"][ei]),float(m["ask"][ei]),eside)
    if not 0<=ubid<uask<=1: row["entry_skip_reason"]="invalid_underdog_bbo"; return row
    fee=quadratic_taker_fee(uask)
    row.update({"entered":True,"entry_skip_reason":None,"underdog_side":uside,"entry_underdog_bid":ubid,
                "entry_underdog_ask":uask,"entry_fee_usd":fee,"entry_cost_usd":uask+fee})
    return row


def settle_underdog(v, side):
    if not np.isfinite(v) or not 0<=v<=1: return None
    return float(v) if side=="YES" else float(1-v)


def simulate_exit(m,a,target_c,max_hold_min):
    ei=int(a["entry_index"]); ets=float(a["entry_ts"]); eask=float(a["entry_underdog_ask"]); uside=str(a["underdog_side"])
    latency_s=float(a["latency_min"])*60.; target=min(1.,eask+target_c/100.); deadline=ets+max_hold_min*60.
    oi=None
    for j in range(ei,len(m["ts"])):
        if m["ts"][j]>deadline+EPS: break
        if _underdog_bid(m["bid"][j],m["ask"][j],uside)+EPS>=target: oi=j; break
    if oi is not None:
        decision="target"; dts=float(m["ts"][oi]); observed=_underdog_bid(m["bid"][oi],m["ask"][oi],uside)
    else:
        decision="timeout"; dts=deadline; observed=np.nan
    desired=dts+latency_s; xi=_first_index_at_or_after(m["ts"],desired)
    payoff=settle_underdog(m["settlement_value"],uside)
    reason=None; xpx=np.nan; xfee=np.nan; xts=np.nan; xlag=np.nan; slip=np.nan
    if xi is not None:
        xpx=_underdog_bid(m["bid"][xi],m["ask"][xi],uside)
        if np.isfinite(xpx) and 0<=xpx<=1:
            xts=float(m["ts"][xi]); xlag=xts-desired; xfee=quadratic_taker_fee(xpx)
            reason="target_executable_bid" if decision=="target" else "timeout_executable_bid"
            if decision=="target": slip=100*(xpx-target)
    if reason is None:
        if payoff is not None:
            reason="settlement_fallback_after_missing_exit_quote"; xpx=payoff; xfee=0.; xts=float(m["ts"][-1])
        else:
            reason="unclosed_missing_exit_quote_and_settlement"
    completed=bool(np.isfinite(xpx) and np.isfinite(xfee)); gross=xpx-eask if completed else np.nan
    net=gross-float(a["entry_fee_usd"])-xfee if completed else np.nan
    endts=xts if np.isfinite(xts) else float(m["ts"][-1]); bids=[]
    for j in range(ei,len(m["ts"])):
        if m["ts"][j]>endts+EPS: break
        bids.append(_underdog_bid(m["bid"][j],m["ask"][j],uside))
    mfe=max(bids)-eask if bids else np.nan; mae=min(bids)-eask if bids else np.nan
    return {**a,"target_c":int(target_c),"max_hold_min":int(max_hold_min),"target_price":target,
            "target_observed":oi is not None,"target_observed_ts":float(m["ts"][oi]) if oi is not None else np.nan,
            "target_observed_bid":observed,"exit_decision_reason":decision,"exit_decision_ts":dts,"desired_exit_ts":desired,
            "exit_reason":reason,"exit_ts":xts,"exit_quote_lag_s":xlag,"exit_underdog_bid":xpx,"exit_fee_usd":xfee,
            "gross_pnl_usd":gross,"net_pnl_usd":net,"gross_pnl_c":100*gross if np.isfinite(gross) else np.nan,
            "net_pnl_c":100*net if np.isfinite(net) else np.nan,
            "return_on_entry_cost":net/float(a["entry_cost_usd"]) if completed and a["entry_cost_usd"]>0 else np.nan,
            "completed":completed,"win":bool(completed and net>0),
            "hold_minutes_actual":(xts-ets)/60 if completed and np.isfinite(xts) else np.nan,
            "target_exec_slippage_c":slip,"mfe_executable_bid_c":100*mfe if np.isfinite(mfe) else np.nan,
            "mae_executable_bid_c":100*mae if np.isfinite(mae) else np.nan}


BASE_RULE_COLS=["spread_cap","latency_min","time_bucket","start_band"]


def _drawdown_usd(g):
    z=g[g.completed.astype(bool)&g.net_pnl_usd.notna()].sort_values(["exit_ts","ticker"])
    if z.empty: return np.nan
    eq=z.net_pnl_usd.cumsum(); peak=eq.cummax().clip(lower=0.0)
    return float((eq-peak).min())


def _peak_concurrency_and_capital(g):
    z=g[g.completed.astype(bool)&g.entry_ts.notna()&g.exit_ts.notna()&g.entry_cost_usd.notna()]
    if z.empty: return 0,0.0
    ev=[]
    for r in z.itertuples(index=False):
        ev.append((float(r.exit_ts),0,-1,-float(r.entry_cost_usd))); ev.append((float(r.entry_ts),1,1,float(r.entry_cost_usd)))
    ev.sort(); n=0; cap=0.; mn=0; mc=0.
    for _,_,dn,dc in ev:
        n+=dn; cap+=dc; mn=max(mn,n); mc=max(mc,cap)
    return int(mn),float(mc)


def summarize_rule_groups(trades,attempts,scope):
    if scope=="regular_or_postseason":
        tr=trades[trades.season_regime==scope].copy(); at=attempts[attempts.season_regime==scope].copy()
    else: tr=trades.copy(); at=attempts.copy()
    rows=[]; tg=["spread_cap","latency_min","time_bucket","start_band","target_c","max_hold_min"]
    amap={k:g for k,g in at.groupby(BASE_RULE_COLS,dropna=False)}
    for keys,g in tr.groupby(tg,dropna=False):
        sp,lat,tb,fb,tc,hm=keys; ag=amap.get((sp,lat,tb,fb),pd.DataFrame()); signals=len(ag); entries=int(ag.entered.astype(bool).sum()) if len(ag) else 0
        c=g[g.completed.astype(bool)].copy(); n=len(c); wins=int(c.win.astype(bool).sum()) if n else 0; wlo,whi=wilson(wins,n); lo,hi=mean_ci(c.net_pnl_c)
        gp=c.loc[c.net_pnl_usd>0,"net_pnl_usd"].sum(); gl=-c.loc[c.net_pnl_usd<0,"net_pnl_usd"].sum(); pf=gp/gl if gl>EPS else (np.inf if gp>0 else np.nan)
        pn,pc=_peak_concurrency_and_capital(c); reasons=Counter(c.exit_reason.astype(str)); skips=Counter(ag.loc[~ag.entered.astype(bool),"entry_skip_reason"].astype(str)) if len(ag) else Counter()
        rows.append({"season_scope":scope,"spread_cap":float(sp),"latency_min":int(lat),"time_bucket":tb,"start_band":fb,"target_c":int(tc),"max_hold_min":int(hm),
            "signal_count":int(signals),"entry_count":entries,"entry_rate":entries/signals if signals else np.nan,"completed_trades":int(n),"unclosed_trades":int(len(g)-n),
            "win_rate":wins/n if n else np.nan,"win_wilson_lo":wlo,"win_wilson_hi":whi,"target_observed_rate":float(c.target_observed.astype(bool).mean()) if n else np.nan,
            "target_exit_rate":reasons["target_executable_bid"]/n if n else np.nan,"timeout_exit_rate":reasons["timeout_executable_bid"]/n if n else np.nan,
            "settlement_fallback_rate":reasons["settlement_fallback_after_missing_exit_quote"]/n if n else np.nan,"mean_gross_pnl_c":float(c.gross_pnl_c.mean()) if n else np.nan,
            "mean_net_pnl_c":float(c.net_pnl_c.mean()) if n else np.nan,"mean_net_pnl_c_ci_lo":lo,"mean_net_pnl_c_ci_hi":hi,"median_net_pnl_c":float(c.net_pnl_c.median()) if n else np.nan,
            "total_q1_net_pnl_usd":float(c.net_pnl_usd.sum()) if n else 0.,"profit_factor":float(pf),"realized_exit_max_drawdown_usd":_drawdown_usd(c),
            "worst_trade_c":float(c.net_pnl_c.min()) if n else np.nan,"best_trade_c":float(c.net_pnl_c.max()) if n else np.nan,
            "mean_hold_min":float(c.hold_minutes_actual.mean()) if n else np.nan,"median_hold_min":float(c.hold_minutes_actual.median()) if n else np.nan,
            "mean_total_fees_c":float(100*(c.entry_fee_usd+c.exit_fee_usd).mean()) if n else np.nan,"mean_return_on_entry_cost":float(c.return_on_entry_cost.mean()) if n else np.nan,
            "mean_mfe_executable_bid_c":float(c.mfe_executable_bid_c.mean()) if n else np.nan,"mean_mae_executable_bid_c":float(c.mae_executable_bid_c.mean()) if n else np.nan,
            "mean_target_exec_slippage_c":float(c.loc[c.target_observed.astype(bool),"target_exec_slippage_c"].mean()) if n else np.nan,
            "max_concurrent_trades":pn,"max_deployed_entry_capital_usd":pc,"skip_no_fresh_quote":skips["no_fresh_quote_after_latency"],
            "skip_side_changed":skips["favorite_side_changed"],"skip_left_band":skips["favorite_left_signal_band"],"skip_spread_widened":skips["spread_widened"]})
    return pd.DataFrame(rows)


def robust_grid_rank(summary):
    q=summary[summary.season_scope=="regular_or_postseason"].copy(); rows=[]; keys=["time_bucket","start_band","target_c","max_hold_min"]
    expected=len(SPREAD_CAPS)*len(LATENCIES_MIN)
    for k,g in q.groupby(keys,dropna=False):
        if len(g)!=expected: continue
        rows.append({**dict(zip(keys,k)),"grid_cells":len(g),"min_completed_trades":int(g.completed_trades.min()),"min_entry_rate":float(g.entry_rate.min()),
            "min_mean_net_pnl_c":float(g.mean_net_pnl_c.min()),"min_mean_net_pnl_c_ci_lo":float(g.mean_net_pnl_c_ci_lo.min()),"median_mean_net_pnl_c":float(g.mean_net_pnl_c.median()),
            "profitable_cells":int((g.mean_net_pnl_c>0).sum()),"positive_ci_cells":int((g.mean_net_pnl_c_ci_lo>0).sum()),
            "worst_realized_drawdown_usd":float(g.realized_exit_max_drawdown_usd.min()),"max_settlement_fallback_rate":float(g.settlement_fallback_rate.max())})
    out=pd.DataFrame(rows)
    if len(out): out=out.sort_values(["positive_ci_cells","profitable_cells","min_mean_net_pnl_c_ci_lo","min_mean_net_pnl_c","min_completed_trades"],ascending=[False,False,False,False,False]).reset_index(drop=True)
    return out


def temporal_detail(trades,robust,top_n=15):
    if robust.empty: return pd.DataFrame()
    q=trades[(trades.season_regime=="regular_or_postseason")&np.isclose(trades.spread_cap,.02)&(trades.latency_min==1)&trades.completed.astype(bool)].copy()
    q["game_month"]=pd.to_datetime(q.game_start_ts,unit="s",utc=True,errors="coerce").dt.strftime("%Y-%m"); rows=[]
    keys=["time_bucket","start_band","target_c","max_hold_min"]
    for _,rr in robust.head(top_n).iterrows():
        mask=(q.time_bucket.astype(str)==str(rr.time_bucket))&(q.start_band.astype(str)==str(rr.start_band))&(pd.to_numeric(q.target_c)==int(rr.target_c))&(pd.to_numeric(q.max_hold_min)==int(rr.max_hold_min))
        g=q[mask].sort_values(["game_start_ts","ticker"]).copy()
        if g.empty: continue
        g["chron_half"]=np.where(np.arange(len(g))<len(g)/2,"early_half","late_half")
        for st,ss in (("chron_half",g.chron_half),("game_month",g.game_month)):
            for val,h in g.groupby(ss,dropna=False):
                lo,hi=mean_ci(h.net_pnl_c); rows.append({**{c:rr[c] for c in keys},"split_type":st,"split_value":str(val),"n":len(h),
                    "mean_net_pnl_c":float(h.net_pnl_c.mean()),"mean_net_pnl_c_ci_lo":lo,"mean_net_pnl_c_ci_hi":hi,"win_rate":float(h.win.astype(bool).mean()),"total_q1_net_pnl_usd":float(h.net_pnl_usd.sum())})
    return pd.DataFrame(rows)


def run(run_dir,show=True):
    static_self_check(show); root=Path(run_dir).expanduser().resolve(); paths,markets=load_inputs(root); mm=build_market_arrays(paths,markets); tickers=sorted(mm)
    attempts_rows=[]; trades_rows=[]; base=[(sp,lat,tb,fb) for sp in SPREAD_CAPS for lat in LATENCIES_MIN for tb in TIME_LABELS for fb in FAV_LABELS]
    if show:
        print("\n"+"="*132); print("CAUSAL Q1 EXECUTABLE GRID"); print("="*132); print("NFL markets:",len(tickers)); print("base rules:",len(base))
    for ri,(sp,lat,tb,fb) in enumerate(base,1):
        for ticker in tickers:
            m=mm[ticker]; a=entry_attempt(m,sp,tb,fb,lat)
            if a is None: continue
            a["game_start_ts"]=float(m["game_start_ts"]); attempts_rows.append(a)
            if not a["entered"]: continue
            for tc in TARGETS_C:
                for hm in MAX_HOLDS_MIN: trades_rows.append(simulate_exit(m,a,tc,hm))
        if show and (ri==len(base) or ri%30==0): print(f"  base rules {ri}/{len(base)} | attempts={len(attempts_rows):,} | trade-rule rows={len(trades_rows):,}")
    attempts=pd.DataFrame(attempts_rows); trades=pd.DataFrame(trades_rows)
    if attempts.empty or trades.empty: raise RuntimeError("No executable backtest rows")
    attempts.to_csv(root/"nfl_exec_entry_attempts.csv.gz",index=False,compression="gzip"); trades.to_csv(root/"nfl_exec_trades.csv.gz",index=False,compression="gzip")
    summary=pd.concat([summarize_rule_groups(trades,attempts,"all"),summarize_rule_groups(trades,attempts,"regular_or_postseason")],ignore_index=True); summary.to_csv(root/"nfl_exec_rule_summary.csv",index=False)
    primary=summary[(summary.season_scope=="regular_or_postseason")&np.isclose(summary.spread_cap,.02)&(summary.latency_min==1)&(summary.completed_trades>=40)].copy()
    primary=primary.sort_values(["mean_net_pnl_c_ci_lo","mean_net_pnl_c","completed_trades"],ascending=[False,False,False]).reset_index(drop=True); primary.to_csv(root/"nfl_exec_primary_rank.csv",index=False)
    robust=robust_grid_rank(summary); robust.to_csv(root/"nfl_exec_robust_grid_rank.csv",index=False); temporal=temporal_detail(trades,robust); temporal.to_csv(root/"nfl_exec_top_temporal.csv",index=False)
    cov=[]
    for k,g in attempts.groupby(BASE_RULE_COLS,dropna=False):
        r=dict(zip(BASE_RULE_COLS,k)); r.update({"signals":len(g),"entries":int(g.entered.astype(bool).sum())}); r["entry_rate"]=r["entries"]/r["signals"] if r["signals"] else np.nan
        for reason,n in Counter(g.loc[~g.entered.astype(bool),"entry_skip_reason"].astype(str)).items(): r[f"skip_{reason}"]=int(n)
        cov.append(r)
    coverage=pd.DataFrame(cov); coverage.to_csv(root/"nfl_exec_entry_coverage.csv",index=False)
    headline={"version":VERSION,"run_dir":str(root),"nfl_markets_with_valid_in_game_bbo":len(tickers),"minute_bbo_source_only":True,"displayed_depth_available":False,
        "quantity_contracts":QTY,"capacity_above_q1_established":False,"fee_model":{"type":"current_deployment_quadratic_taker","formula":"ceil_0.0001(0.07 * multiplier * qty * p * (1-p))","multiplier":TAKER_MULTIPLIER,"historical_2025_fee_schedule_reconstructed":False},
        "spread_caps":list(SPREAD_CAPS),"latencies_min":list(LATENCIES_MIN),"time_buckets":list(TIME_LABELS),"favorite_bands":list(FAV_LABELS),"targets_c":list(TARGETS_C),"max_holds_min":list(MAX_HOLDS_MIN),
        "max_quote_lag_s":MAX_QUOTE_LAG_S,"signal_attempt_rows":len(attempts),"entered_base_attempts":int(attempts.entered.astype(bool).sum()),"trade_rule_rows":len(trades),
        "completed_trade_rule_rows":int(trades.completed.astype(bool).sum()),"primary_rules_n_ge_40":len(primary),"robust_rule_families":len(robust),"orders_sent":False,"api_called":False,
        "scientific_guardrail":"In-sample candidate-development economics on one-minute BBO closes; not alpha proof, sub-minute fill proof, or capacity evidence above Q1. Primary uses 1-minute latency; 0-minute is optimistic sensitivity only."}
    _atomic_json(root/"nfl_exec_headline.json",headline)
    if show:
        print("\n"+"="*132); print("PRIMARY RANK — REGULAR/POSTSEASON, SPREAD <=2c, LATENCY 1m, >=40 COMPLETED"); print("="*132)
        cols=["time_bucket","start_band","target_c","max_hold_min","signal_count","entry_count","entry_rate","completed_trades","win_rate","target_exit_rate","mean_net_pnl_c","mean_net_pnl_c_ci_lo","mean_net_pnl_c_ci_hi","total_q1_net_pnl_usd","profit_factor","realized_exit_max_drawdown_usd","median_hold_min","mean_total_fees_c"]
        print(primary[cols].head(40).to_string(index=False) if len(primary) else "none")
        print("\n"+"="*132); print("ROBUST FAMILY RANK — REGULAR/POSTSEASON ACROSS 1c/2c × LATENCY 0/1/2"); print("="*132); print(robust.head(40).to_string(index=False) if len(robust) else "none")
        print("\n"+"="*132); print("ENTRY COVERAGE — PRIMARY 2c / 1m"); print("="*132); pc=coverage[np.isclose(coverage.spread_cap,.02)&(coverage.latency_min==1)]; print(pc.to_string(index=False))
        print("\nIMPORTANT LIMITATIONS\n- one-minute BBO CLOSES, not tick BBO\n- no displayed size/depth => Q1 only; no capacity claim\n- current KXNFLGAME fee model applied; historical 2025 fee schedule not reconstructed\n- zero-minute latency is optimistic sensitivity; primary uses one minute\n- sample already used for candidate discovery; NOT independent OOS")
        print("\nOutput:",root)
    return headline


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); a=ap.parse_args(); run(a.run_dir,show=True)


if __name__=="__main__": main()

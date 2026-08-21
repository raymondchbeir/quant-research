from __future__ import annotations

"""Account-rounding hardening for the causal NFL executable backtest V1.

Runs V1's causal Q1 simulation, then recomputes economics using Kalshi's documented
single-order balance-rounding mechanics in addition to the centicent trade fee.
V1 trade-fee-only outputs are preserved as a baseline. Historical depth/fill
fragmentation is unavailable, so each separate Q1 marketable order is modeled as
one aggregate fill for account rounding. No API calls, orders, or live-Q50 imports.
"""

import argparse, importlib.util, math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "KALSHI_SPORTS_NFL_EXECUTABLE_BACKTEST_V1_1_ACCOUNT_ROUNDED_Q1"


def _load_v1():
    path = Path(__file__).with_name("kalshi_sports_gap_nfl_executable_backtest_v1.py")
    spec = importlib.util.spec_from_file_location("_nfl_exec_v1", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


V1 = _load_v1()


def single_fill_order_fee(price: float, action: str):
    p=float(price); trade_fee=float(V1.quadratic_taker_fee(p))
    revenue=-p if action=="BUY" else p if action=="SELL" else None
    if revenue is None: raise ValueError(action)
    balance_change=revenue-trade_fee
    posted_change=math.floor(balance_change*100.0+1e-12)/100.0
    rounding_fee=max(0.0,balance_change-posted_change)
    return trade_fee,rounding_fee,trade_fee+rounding_fee


def static_self_check(show=True):
    b50=single_fill_order_fee(.50,"BUY"); s50=single_fill_order_fee(.50,"SELL"); b20=single_fill_order_fee(.20,"BUY")
    out={"version":VERSION,"offline_only":True,"api_called":False,"orders_sent":False,"live_q50_modules_imported":False,
         "base_simulator":V1.VERSION,"q1_only":True,"separate_market_order_per_entry_exit":True,
         "trade_fee_centicent_rounding":True,"balance_rounding_mode":"single_Q1_aggregate_fill_per_order",
         "fee_50_buy_net":b50[2],"fee_50_sell_net":s50[2],"fee_20_buy_net":b20[2]}
    out["ok"]=(abs(b50[0]-.0175)<1e-12 and abs(b50[2]-.02)<1e-12 and abs(s50[2]-.02)<1e-12 and abs(b20[0]-.0112)<1e-12 and abs(b20[2]-.02)<1e-12)
    if show:
        print("="*132); print("NFL EXECUTABLE BACKTEST V1.1 — ACCOUNT ROUNDING HARDENING"); print("="*132)
        for k,v in out.items(): print(f"{k:56s}: {v}")
    if not out["ok"]: raise RuntimeError("V1.1 static self-check failed")
    return out


def harden_attempts(attempts):
    x=attempts.copy(); x["entry_trade_fee_usd"]=np.nan; x["entry_balance_rounding_fee_usd"]=np.nan
    for i in x.index[x.entered.astype(bool)]:
        tf,rf,nf=single_fill_order_fee(float(x.at[i,"entry_underdog_ask"]),"BUY")
        x.at[i,"entry_trade_fee_usd"]=tf; x.at[i,"entry_balance_rounding_fee_usd"]=rf; x.at[i,"entry_fee_usd"]=nf; x.at[i,"entry_cost_usd"]=float(x.at[i,"entry_underdog_ask"])+nf
    return x


def harden_trades(trades):
    x=trades.copy()
    for c in ("entry_trade_fee_usd","entry_balance_rounding_fee_usd","exit_trade_fee_usd","exit_balance_rounding_fee_usd"): x[c]=np.nan
    for i in x.index:
        if not bool(x.at[i,"entered"]): continue
        ep=float(x.at[i,"entry_underdog_ask"]); etf,erf,enf=single_fill_order_fee(ep,"BUY")
        x.at[i,"entry_trade_fee_usd"]=etf; x.at[i,"entry_balance_rounding_fee_usd"]=erf; x.at[i,"entry_fee_usd"]=enf; x.at[i,"entry_cost_usd"]=ep+enf
        if not bool(x.at[i,"completed"]): continue
        reason=str(x.at[i,"exit_reason"]); xp=float(x.at[i,"exit_underdog_bid"])
        if reason=="settlement_fallback_after_missing_exit_quote": xtf=xrf=xnf=0.0
        else: xtf,xrf,xnf=single_fill_order_fee(xp,"SELL")
        x.at[i,"exit_trade_fee_usd"]=xtf; x.at[i,"exit_balance_rounding_fee_usd"]=xrf; x.at[i,"exit_fee_usd"]=xnf
        gross=xp-ep; net=gross-enf-xnf
        x.at[i,"gross_pnl_usd"]=gross; x.at[i,"gross_pnl_c"]=100*gross; x.at[i,"net_pnl_usd"]=net; x.at[i,"net_pnl_c"]=100*net
        x.at[i,"return_on_entry_cost"]=net/(ep+enf) if ep+enf>0 else np.nan; x.at[i,"win"]=bool(net>0)
    x["total_trade_fee_usd"]=x.entry_trade_fee_usd.fillna(0)+x.exit_trade_fee_usd.fillna(0)
    x["total_balance_rounding_fee_usd"]=x.entry_balance_rounding_fee_usd.fillna(0)+x.exit_balance_rounding_fee_usd.fillna(0)
    x["total_effective_fee_usd"]=x.entry_fee_usd.fillna(0)+x.exit_fee_usd.fillna(0)
    return x


def coverage_table(attempts):
    rows=[]
    for keys,g in attempts.groupby(V1.BASE_RULE_COLS,dropna=False):
        row=dict(zip(V1.BASE_RULE_COLS,keys)); row["signals"]=len(g); row["entries"]=int(g.entered.astype(bool).sum()); row["entry_rate"]=row["entries"]/row["signals"] if row["signals"] else np.nan
        for reason,n in Counter(g.loc[~g.entered.astype(bool),"entry_skip_reason"].astype(str)).items(): row[f"skip_{reason}"]=int(n)
        rows.append(row)
    return pd.DataFrame(rows)


def run(run_dir,show=True):
    static_self_check(show); root=Path(run_dir).expanduser().resolve()
    if show: print("\nRunning causal V1 path/execution simulation first...")
    baseline=V1.run(root,show=False)
    attempts=harden_attempts(pd.read_csv(root/"nfl_exec_entry_attempts.csv.gz",compression="gzip"))
    trades=harden_trades(pd.read_csv(root/"nfl_exec_trades.csv.gz",compression="gzip"))
    attempts.to_csv(root/"nfl_exec_v11_entry_attempts.csv.gz",index=False,compression="gzip"); trades.to_csv(root/"nfl_exec_v11_trades.csv.gz",index=False,compression="gzip")
    summary=pd.concat([V1.summarize_rule_groups(trades,attempts,"all"),V1.summarize_rule_groups(trades,attempts,"regular_or_postseason")],ignore_index=True); summary.to_csv(root/"nfl_exec_v11_rule_summary.csv",index=False)
    primary=summary[(summary.season_scope=="regular_or_postseason")&np.isclose(summary.spread_cap,.02)&(summary.latency_min==1)&(summary.completed_trades>=40)].copy().sort_values(["mean_net_pnl_c_ci_lo","mean_net_pnl_c","completed_trades"],ascending=[False,False,False]).reset_index(drop=True); primary.to_csv(root/"nfl_exec_v11_primary_rank.csv",index=False)
    robust=V1.robust_grid_rank(summary); robust.to_csv(root/"nfl_exec_v11_robust_grid_rank.csv",index=False)
    temporal=V1.temporal_detail(trades,robust,top_n=15); temporal.to_csv(root/"nfl_exec_v11_top_temporal.csv",index=False)
    coverage=coverage_table(attempts); coverage.to_csv(root/"nfl_exec_v11_entry_coverage.csv",index=False)
    c=trades[trades.completed.astype(bool)].copy()
    headline={"version":VERSION,"base_simulator_version":baseline.get("version"),"run_dir":str(root),"q1_only":True,"minute_bbo_close_only":True,"displayed_depth_available":False,"capacity_above_q1_established":False,
              "current_deployment_fee_model":{"trade_fee":"ceil_0.0001(0.07 * 1 * qty * p * (1-p))","balance_rounding":"single Q1 aggregate fill per separate marketable order","fee_accumulator_across_partial_fills_observable":False,"historical_2025_fee_schedule_reconstructed":False},
              "completed_trade_rule_rows":len(c),"mean_trade_fee_only_component_c":float(100*c.total_trade_fee_usd.mean()) if len(c) else None,"mean_balance_rounding_component_c":float(100*c.total_balance_rounding_fee_usd.mean()) if len(c) else None,"mean_total_effective_fee_c":float(100*c.total_effective_fee_usd.mean()) if len(c) else None,"primary_rules_n_ge_40":len(primary),"robust_rule_families":len(robust),"api_called":False,"orders_sent":False,
              "guardrail":"Candidate-development only. One-minute BBO closes cannot establish sub-minute fill certainty or capacity. Q1 fill decomposition is absent; account rounding uses one aggregate fill per separate marketable order. Current deployment fees are applied to historical paths; historical fee schedule is not reconstructed."}
    V1._atomic_json(root/"nfl_exec_v11_headline.json",headline)
    if show:
        print("\n"+"="*132); print("PRIMARY ACCOUNT-ROUNDED RANK — REGULAR/POSTSEASON, <=2c, 1m LATENCY, N>=40"); print("="*132)
        cols=["time_bucket","start_band","target_c","max_hold_min","signal_count","entry_count","entry_rate","completed_trades","win_rate","target_exit_rate","mean_net_pnl_c","mean_net_pnl_c_ci_lo","mean_net_pnl_c_ci_hi","total_q1_net_pnl_usd","profit_factor","realized_exit_max_drawdown_usd","median_hold_min","mean_total_fees_c"]
        print(primary[cols].head(50).to_string(index=False) if len(primary) else "none")
        print("\n"+"="*132); print("ROBUST ACCOUNT-ROUNDED FAMILIES — ALL 1c/2c x LATENCY 0/1/2"); print("="*132); print(robust.head(50).to_string(index=False) if len(robust) else "none")
        print("\n"+"="*132); print("FEE HARDENING"); print("="*132)
        for k in ("mean_trade_fee_only_component_c","mean_balance_rounding_component_c","mean_total_effective_fee_c"): print(f"{k:42s}: {headline[k]}")
        print("\nIMPORTANT:\n- V1 trade-fee-only files preserved for audit.\n- V1.1 account-rounded files are PRIMARY economics.\n- Q1 only: sports history has no displayed depth.\n- 1-minute latency primary; zero-minute optimistic sensitivity.\n- Still in-sample candidate development, not independent OOS.")
        print("\nOutput:",root)
    return headline


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); a=ap.parse_args(); run(a.run_dir,show=True)


if __name__=="__main__": main()

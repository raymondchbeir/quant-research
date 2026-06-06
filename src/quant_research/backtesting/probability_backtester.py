from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.utils.paths import PROCESSED_DATA_DIR


def load_market_data(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """
    Load the full featured market dataset.

    We use this file because it still contains NVDA open/high/low/close,
    while the model feature file only contains model-safe columns.
    """
    path = Path(processed_dir) / f"featured_{timeframe}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing featured dataset: {path}")

    df = pd.read_parquet(path)

    required_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "NVDA_open",
        "NVDA_high",
        "NVDA_low",
        "NVDA_close",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Market data is missing required columns:\n"
            f"{missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def load_validation_predictions(results_dir: Path) -> pd.DataFrame:
    """
    Load all fold validation prediction files from one model experiment.
    """
    prediction_paths = sorted(
        Path(results_dir).glob("fold_*/validation_predictions.parquet")
    )

    if not prediction_paths:
        raise FileNotFoundError(
            f"No validation prediction files found in {results_dir}"
        )

    prediction_dfs = []

    for path in prediction_paths:
        fold_name = path.parent.name

        pred = pd.read_parquet(path)
        pred["fold"] = fold_name

        prediction_dfs.append(pred)

    predictions = pd.concat(
        prediction_dfs,
        axis=0,
        ignore_index=True,
    )

    required_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "target_name",
        "target_class",
        "p_neutral",
        "p_up",
        "p_down",
        "fold",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Prediction files are missing columns:\n{missing_columns}"
        )

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"],
        utc=True,
    )
    predictions["timestamp_pt"] = pd.to_datetime(
        predictions["timestamp_pt"]
    )
    predictions["date"] = predictions["timestamp_pt"].dt.date

    predictions = predictions.sort_values(
        ["fold", "timestamp"]
    ).reset_index(drop=True)

    return predictions


def apply_signal_rule(
    predictions: pd.DataFrame,
    edge_threshold: float,
    min_direction_probability: float,
    long_edge_threshold: float | None = None,
    short_edge_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Create long/short/flat signals from model probabilities.

    If long_edge_threshold and short_edge_threshold are not provided,
    both sides use edge_threshold.

    Long:
        P(up) - P(down) >= long_edge_threshold
        and P(up) >= min_direction_probability

    Short:
        P(down) - P(up) >= short_edge_threshold
        and P(down) >= min_direction_probability
    """
    df = predictions.copy()

    if long_edge_threshold is None:
        long_edge_threshold = edge_threshold

    if short_edge_threshold is None:
        short_edge_threshold = edge_threshold

    up_edge = df["p_up"] - df["p_down"]
    down_edge = df["p_down"] - df["p_up"]

    long_mask = (
        (up_edge >= long_edge_threshold)
        & (df["p_up"] >= min_direction_probability)
    )

    short_mask = (
        (down_edge >= short_edge_threshold)
        & (df["p_down"] >= min_direction_probability)
    )

    df["signal"] = "flat"
    df.loc[long_mask, "signal"] = "long"
    df.loc[short_mask, "signal"] = "short"

    df["signal_edge"] = np.where(
        df["signal"] == "long",
        up_edge,
        np.where(
            df["signal"] == "short",
            down_edge,
            0.0,
        ),
    )

    df["long_edge_threshold"] = long_edge_threshold
    df["short_edge_threshold"] = short_edge_threshold

    return df

def build_market_lookup(market_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build fast lookup dictionaries.

    timestamp_to_index:
        timestamp -> row index in market_df

    date_to_indices:
        date -> numpy array of row indices for that date
    """
    timestamp_to_index = {
        timestamp: index
        for index, timestamp in enumerate(market_df["timestamp"])
    }

    date_to_indices = {
        date: group.index.to_numpy()
        for date, group in market_df.groupby("date", sort=False)
    }

    return timestamp_to_index, date_to_indices


def simulate_one_trade(
    market_df: pd.DataFrame,
    signal_row: pd.Series,
    signal_index: int,
    timestamp_to_index: dict,
    date_to_indices: dict,
    horizon_bars: int = 12,
    take_profit_pct: float = 0.01,
    stop_loss_pct: float = 0.01,
    slippage_bps: float = 1.0,
    fee_bps_per_side: float = 0.0,
) -> dict | None:
    """
    Simulate one long or short trade.

    Signal is generated after candle t closes.
    Entry happens at candle t+1 open.

    Conservative same-bar rule:
        If take profit and stop loss are both touched in the same candle,
        assume the stop loss happened first.
    """
    signal = signal_row["signal"]

    if signal not in ["long", "short"]:
        return None

    timestamp = signal_row["timestamp"]
    trade_date = signal_row["date"]

    if timestamp not in timestamp_to_index:
        return None

    signal_market_index = timestamp_to_index[timestamp]

    day_indices = date_to_indices.get(trade_date)

    if day_indices is None:
        return None

    day_indices_set = set(day_indices)

    entry_index = signal_market_index + 1

    if entry_index not in day_indices_set:
        return None

    raw_entry_price = float(
        market_df.loc[entry_index, "NVDA_open"]
    )

    if raw_entry_price <= 0:
        return None

    slippage = slippage_bps / 10_000
    fee_per_side = fee_bps_per_side / 10_000

    # Execution prices include adverse slippage.
    if signal == "long":
        entry_price = raw_entry_price * (1 + slippage)
        take_profit_price = raw_entry_price * (1 + take_profit_pct)
        stop_loss_price = raw_entry_price * (1 - stop_loss_pct)
    else:
        entry_price = raw_entry_price * (1 - slippage)
        take_profit_price = raw_entry_price * (1 - take_profit_pct)
        stop_loss_price = raw_entry_price * (1 + stop_loss_pct)

    max_exit_index = min(
        entry_index + horizon_bars - 1,
        int(day_indices[-1]),
    )

    future_indices = range(entry_index, max_exit_index + 1)

    exit_index = max_exit_index
    raw_exit_price = float(
        market_df.loc[max_exit_index, "NVDA_close"]
    )
    exit_reason = "timeout"

    for current_index in future_indices:
        high_price = float(market_df.loc[current_index, "NVDA_high"])
        low_price = float(market_df.loc[current_index, "NVDA_low"])

        if signal == "long":
            take_profit_hit = high_price >= take_profit_price
            stop_loss_hit = low_price <= stop_loss_price

            if take_profit_hit and stop_loss_hit:
                exit_index = current_index
                raw_exit_price = stop_loss_price
                exit_reason = "same_bar_stop_loss"
                break

            if stop_loss_hit:
                exit_index = current_index
                raw_exit_price = stop_loss_price
                exit_reason = "stop_loss"
                break

            if take_profit_hit:
                exit_index = current_index
                raw_exit_price = take_profit_price
                exit_reason = "take_profit"
                break

        else:
            take_profit_hit = low_price <= take_profit_price
            stop_loss_hit = high_price >= stop_loss_price

            if take_profit_hit and stop_loss_hit:
                exit_index = current_index
                raw_exit_price = stop_loss_price
                exit_reason = "same_bar_stop_loss"
                break

            if stop_loss_hit:
                exit_index = current_index
                raw_exit_price = stop_loss_price
                exit_reason = "stop_loss"
                break

            if take_profit_hit:
                exit_index = current_index
                raw_exit_price = take_profit_price
                exit_reason = "take_profit"
                break

    # If horizon was cut off by end of day, label it clearly.
    if exit_reason == "timeout" and exit_index == int(day_indices[-1]):
        exit_reason = "market_close_or_day_end"

    if signal == "long":
        exit_price = raw_exit_price * (1 - slippage)
        gross_return = (exit_price / entry_price) - 1
    else:
        exit_price = raw_exit_price * (1 + slippage)
        gross_return = (entry_price / exit_price) - 1

    net_return = gross_return - (2 * fee_per_side)

    return {
        "fold": signal_row["fold"],
        "date": trade_date,
        "signal_timestamp": signal_row["timestamp"],
        "signal_timestamp_pt": signal_row["timestamp_pt"],
        "entry_timestamp": market_df.loc[entry_index, "timestamp"],
        "entry_timestamp_pt": market_df.loc[entry_index, "timestamp_pt"],
        "exit_timestamp": market_df.loc[exit_index, "timestamp"],
        "exit_timestamp_pt": market_df.loc[exit_index, "timestamp_pt"],
        "signal_index": signal_market_index,
        "entry_index": entry_index,
        "exit_index": exit_index,
        "side": signal,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "raw_entry_price": raw_entry_price,
        "raw_exit_price": raw_exit_price,
        "gross_return": gross_return,
        "net_return": net_return,
        "exit_reason": exit_reason,
        "p_neutral": signal_row["p_neutral"],
        "p_up": signal_row["p_up"],
        "p_down": signal_row["p_down"],
        "signal_edge": signal_row["signal_edge"],
        "target_name": signal_row["target_name"],
        "target_class": signal_row["target_class"],
        "bars_held": exit_index - entry_index + 1,
    }


def run_backtest_for_threshold(
    market_df: pd.DataFrame,
    predictions: pd.DataFrame,
    edge_threshold: float,
    min_direction_probability: float,
    horizon_bars: int = 12,
    take_profit_pct: float = 0.01,
    stop_loss_pct: float = 0.01,
    slippage_bps: float = 1.0,
    fee_bps_per_side: float = 0.0,
    long_edge_threshold: float | None = None,
    short_edge_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Run a non-overlapping trade backtest for one threshold setting.

    If already in a trade, ignore new signals until that trade exits.
    """
    signal_df = apply_signal_rule(
        predictions=predictions,
        edge_threshold=edge_threshold,
        min_direction_probability=min_direction_probability,
        long_edge_threshold=long_edge_threshold,
        short_edge_threshold=short_edge_threshold,
    )

    signal_df = signal_df.loc[
        signal_df["signal"].isin(["long", "short"])
    ].copy()

    if signal_df.empty:
        return pd.DataFrame()

    timestamp_to_index, date_to_indices = build_market_lookup(market_df)

    trades = []

    # Track active trades by fold/date so trades do not overlap.
    last_exit_index_by_fold_date = {}

    signal_df = signal_df.sort_values(
        ["fold", "timestamp"]
    ).reset_index(drop=True)

    for _, signal_row in signal_df.iterrows():
        fold = signal_row["fold"]
        trade_date = signal_row["date"]

        key = (fold, trade_date)

        timestamp = signal_row["timestamp"]

        if timestamp not in timestamp_to_index:
            continue

        signal_market_index = timestamp_to_index[timestamp]

        last_exit_index = last_exit_index_by_fold_date.get(key, -1)

        if signal_market_index <= last_exit_index:
            continue

        trade = simulate_one_trade(
            market_df=market_df,
            signal_row=signal_row,
            signal_index=signal_market_index,
            timestamp_to_index=timestamp_to_index,
            date_to_indices=date_to_indices,
            horizon_bars=horizon_bars,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            slippage_bps=slippage_bps,
            fee_bps_per_side=fee_bps_per_side,
        )

        if trade is None:
            continue

        trades.append(trade)
        last_exit_index_by_fold_date[key] = trade["exit_index"]

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df["edge_threshold"] = edge_threshold
        trades_df["min_direction_probability"] = (
            min_direction_probability
        )
        trades_df["horizon_bars"] = horizon_bars
        trades_df["take_profit_pct"] = take_profit_pct
        trades_df["stop_loss_pct"] = stop_loss_pct
        trades_df["slippage_bps"] = slippage_bps
        trades_df["fee_bps_per_side"] = fee_bps_per_side

    return trades_df


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate max drawdown from sequential trade returns.
    """
    if returns.empty:
        return np.nan

    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1

    return float(drawdown.min())


def summarize_trades(trades_df: pd.DataFrame) -> dict:
    """
    Summarize one backtest result.
    """
    if trades_df.empty:
        return {
            "trade_count": 0,
            "long_trades": 0,
            "short_trades": 0,
            "win_rate": np.nan,
            "average_return": np.nan,
            "median_return": np.nan,
            "average_win": np.nan,
            "average_loss": np.nan,
            "profit_factor": np.nan,
            "total_net_return_compounded": np.nan,
            "total_net_return_sum": np.nan,
            "max_drawdown": np.nan,
        }

    returns = trades_df["net_return"]

    wins = returns.loc[returns > 0]
    losses = returns.loc[returns < 0]

    gross_profit = wins.sum()
    gross_loss = -losses.sum()

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.nan
    )

    compounded_return = (1 + returns).prod() - 1

    return {
        "trade_count": int(len(trades_df)),
        "long_trades": int((trades_df["side"] == "long").sum()),
        "short_trades": int((trades_df["side"] == "short").sum()),
        "win_rate": float((returns > 0).mean()),
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "average_win": float(wins.mean()) if len(wins) else np.nan,
        "average_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(profit_factor),
        "total_net_return_compounded": float(compounded_return),
        "total_net_return_sum": float(returns.sum()),
        "max_drawdown": calculate_max_drawdown(returns),
    }


def run_probability_backtest_sweep(
    experiment_name: str,
    results_dir: Path,
    output_dir: Path,
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    edge_thresholds: list[float] | None = None,
    min_direction_probability: float = 0.50,
    horizon_bars: int = 12,
    take_profit_pct: float = 0.01,
    stop_loss_pct: float = 0.01,
    slippage_bps: float = 1.0,
    fee_bps_per_side: float = 0.0,
    long_edge_threshold: float | None = None,
    short_edge_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Run a backtest sweep over probability edge thresholds.
    """
    if edge_thresholds is None:
        edge_thresholds = [0.05, 0.10, 0.15]

    market_df = load_market_data(
        timeframe=timeframe,
        processed_dir=processed_dir,
    )

    predictions = load_validation_predictions(results_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for edge_threshold in edge_thresholds:
        trades_df = run_backtest_for_threshold(
            market_df=market_df,
            predictions=predictions,
            edge_threshold=edge_threshold,
            min_direction_probability=min_direction_probability,
            horizon_bars=horizon_bars,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            slippage_bps=slippage_bps,
            fee_bps_per_side=fee_bps_per_side,
            long_edge_threshold=long_edge_threshold,
            short_edge_threshold=short_edge_threshold,
        )

        threshold_name = (
            f"edge_{edge_threshold:.3f}"
            .replace(".", "p")
        )

        trades_path = output_dir / f"trades_{threshold_name}.parquet"

        trades_df.to_parquet(trades_path, index=False)

        summary = summarize_trades(trades_df)

        summary.update(
            {
                "experiment_name": experiment_name,
                "edge_threshold": edge_threshold,
                "min_direction_probability": (
                    min_direction_probability
                ),
                "horizon_bars": horizon_bars,
                "take_profit_pct": take_profit_pct,
                "stop_loss_pct": stop_loss_pct,
                "slippage_bps": slippage_bps,
                "fee_bps_per_side": fee_bps_per_side,
                "trades_path": str(trades_path),
                "long_edge_threshold": long_edge_threshold,
                "short_edge_threshold": short_edge_threshold,
            }
        )

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = output_dir / "backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Experiment: {experiment_name}")
    print(f"Saved backtest summary to: {summary_path}")
    print("\nBacktest summary:")
    print(
        summary_df[
            [
                "edge_threshold",
                "trade_count",
                "long_trades",
                "short_trades",
                "win_rate",
                "average_return",
                "average_win",
                "average_loss",
                "profit_factor",
                "total_net_return_compounded",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )

    return summary_df


def run_default_probability_backtests(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Run first-pass backtests for the main XGBoost experiments.
    """
    processed_dir = Path(processed_dir)

    experiments = {
        "lagged_unweighted": (
            processed_dir
            / f"xgboost_walk_forward_{timeframe}"
        ),
        "lagged_balanced": (
            processed_dir
            / f"xgboost_walk_forward_{timeframe}_balanced"
        ),
    }

    outputs = {}

    for experiment_name, results_dir in experiments.items():
        if not results_dir.exists():
            print(
                f"Skipping {experiment_name}: "
                f"missing {results_dir}"
            )
            continue

        output_dir = (
            processed_dir
            / f"probability_backtests_{timeframe}"
            / experiment_name
        )

        outputs[experiment_name] = run_probability_backtest_sweep(
            experiment_name=experiment_name,
            results_dir=results_dir,
            output_dir=output_dir,
            timeframe=timeframe,
            processed_dir=processed_dir, 
            edge_thresholds=[0.05, 0.10, 0.15],
            min_direction_probability=0.50,
            horizon_bars=12,
            take_profit_pct=0.99,
            stop_loss_pct=0.99,
            slippage_bps=1.0,
            fee_bps_per_side=0.0,
        )

    return outputs
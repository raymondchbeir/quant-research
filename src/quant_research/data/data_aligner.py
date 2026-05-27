from pathlib import Path
import pandas as pd
from quant_research.utils.paths import PROCESSED_DATA_DIR

def load_processed_symbol_data(symbol, timeframe="5min", processed_dir=PROCESSED_DATA_DIR):
    """
    Load processed Alpaca parquet data for one symbol.
    """
    # Makes the path per symbol
    path = Path(processed_dir) / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data for {symbol}: {path}")
    
    # Reading the data
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{symbol}: missing timestamp column.")
    
    #changing the timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    #returning the dataframe
    return df

def prefix_symbol_columns(df, symbol):
    rename_map = {
    "open": f"{symbol}_open",
    "high": f"{symbol}_high",
    "low": f"{symbol}_low",
    "close": f"{symbol}_close",
    "volume": f"{symbol}_volume",
    "trade_count": f"{symbol}_trade_count",
    "vwap": f"{symbol}_vwap",
    }
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    df = df.rename(columns = rename_map).copy()
    return df

def build_aligned_dataset(symbols, timeframe="5min", processed_dir=PROCESSED_DATA_DIR):
    """
    Load processed per-symbol parquet files, prefix each symbol's columns,
    merge everything on timestamp, and save one aligned parquet file.

    Output:
        data/processed/aligned_5min.parquet
    """
    aligned_df = None
    for i, symbol in enumerate(symbols):
        print(f"Loading and aligning {symbol}...")
        df = load_processed_symbol_data(
            symbol=symbol,
            timeframe=timeframe,
            processed_dir=processed_dir,
            )
        df = prefix_symbol_columns(df, symbol)
        if i == 0:
            aligned_df = df.copy()
        else:
            # These columns are already kept from the first symbol
            df = df.drop(
                columns=["timestamp_pt", "date", "time"],
                errors="ignore",
            )
            aligned_df = aligned_df.merge(
                df,
                on="timestamp",
                how="inner",
            )
    aligned_df = aligned_df.sort_values("timestamp").reset_index(drop=True)
    output_path = Path(processed_dir) / f"aligned_{timeframe}.parquet"
    aligned_df.to_parquet(output_path, index=False)
    print(f"Saved aligned dataset: {len(aligned_df):,} rows -> {output_path}")
    return aligned_df
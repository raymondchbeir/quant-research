from pathlib import Path
from datetime import time
from quant_research.data.data_downloader import confirm_file_action
import pandas as pd

from quant_research.utils.paths import (
    ALPACA_RAW_DIR,
    PROCESSED_DATA_DIR,
)

# Market session settings in Pacific Time
MARKET_TIMEZONE = "America/Los_Angeles"
# Regular market open and close
MARKET_OPEN = time(6, 30)
MARKET_CLOSE = time(13, 0)
# We want 21 candles before open for EMA warmup
WARMUP_CANDLES = 21
# 5-minute candles
CANDLE_MINUTES = 5
# 21 candles * 5 minutes = 105 minutes
WARMUP_MINUTES = WARMUP_CANDLES * CANDLE_MINUTES
# Expected rows per symbol per day:
# 21 warmup candles + 78 regular session candles = 99 candles
EXPECTED_CANDLES_PER_DAY_WITH_WARMUP = 21 + 78

def load_raw_symbol_data(symbol, timeframe = "5min", raw_dir = ALPACA_RAW_DIR):
    """
    Load raw Alpaca parquet data for one symbol.
    """
    # Makes the path per symbol
    path = Path(raw_dir) / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data for {symbol}: {path}")
    
    # Reading the data
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{symbol}: missing timestamp column.")
    
    #changing the timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    #returning the dataframe
    return df

def save_overlapping_days(symbols):
    # Defining the thresholds for checking if a candle is in between those times
    market_open_minutes = 6 * 60 + 30
    market_close_minutes = 13 * 60
    warmup_start_minutes = market_open_minutes - 21 * 5
    #defining what will store the dfs

    cleaned_dfs = {}
    complete_days_by_symbol = {}
    for symbol in symbols:
        df = load_raw_symbol_data(symbol)
        df["timestamp_pt"] = df["timestamp"].dt.tz_convert("America/Los_Angeles")
        #creating the date column
        df["date"] = df["timestamp_pt"].dt.date
        #creating the time Column
        df["time"] = df["timestamp_pt"].dt.time

        print(f"Transformed {symbol} to pacific time")

        # 
        minutes_from_midnight = ( df["timestamp_pt"].dt.hour * 60 + df["timestamp_pt"].dt.minute)


        #this drops all candles outside the time
        df = df[(minutes_from_midnight >= warmup_start_minutes)
                & (minutes_from_midnight < market_close_minutes)].copy()
        print("outside candles removed")

        #this outputs a pandas series with the size of each day
        day_counts = df.groupby("date").size()

        #keeps the dates that have a day count of 99
        complete_days = day_counts[day_counts == 99].index
        print(f"{symbol} has {day_counts.size} total days after time filtering")
        print(f"{symbol} has {len(complete_days)} complete days")

        #returns a clean df
        df_clean = df[df["date"].isin(complete_days)].copy()
        cleaned_dfs[symbol] = df_clean
        complete_days_by_symbol[symbol] = set(complete_days)
    common_days = set.intersection(*complete_days_by_symbol.values())
    print(f"initial cleaning complete, there are {len(common_days)} in common")
    for symbol in symbols:
        final_df = cleaned_dfs[symbol][cleaned_dfs[symbol]["date"].isin(common_days)].copy() 
        final_df = final_df.sort_values("timestamp").reset_index(drop=True)
        if confirm_file_action(symbol, PROCESSED_DATA_DIR, timeframe="5min", action_name="save processed data"):
            output_path = PROCESSED_DATA_DIR / f"{symbol}_5min.parquet"
            final_df.to_parquet(output_path, index=False)
            print(f"Saved processed {symbol}: {len(final_df):,} rows -> {output_path}")
        else:
            print(f"Skipped saving processed {symbol}.")

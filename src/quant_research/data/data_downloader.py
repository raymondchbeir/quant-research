import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockLatestTradeRequest

from quant_research.utils.paths import (
    PROJECT_ROOT,
    ALPACA_RAW_DIR,
)


load_dotenv(PROJECT_ROOT / ".env")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED")
client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)


#check for a connection
def CheckConnection():
    request = StockLatestTradeRequest(
    symbol_or_symbols=["NVDA"],
    feed=DataFeed.IEX,
    )

    try:
        response = client.get_stock_latest_trade(request)
        trade = response["NVDA"]
        print("Alpaca connected")
    except Exception as e:
        print("Alpaca connection failed")
        print(type(e).__name__)
        print(e)

# Save the data in a parquet
def save_symbol_parquets(df, output_dir, timeframe):
    output_dir.mkdir(parents=True, exist_ok=True)
    required_columns = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    #checking for missing columns 
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    for symbol, symbol_df in df.groupby("symbol"):
        #making the index the symbol by itself
        symbol_df = symbol_df.sort_values("timestamp").reset_index(drop=True)
        output_path = output_dir / f"{symbol}_{timeframe}.parquet"
        symbol_df.to_parquet(output_path, index=False)

        print(f"Saved {symbol}: {len(symbol_df):,} rows -> {output_path}")

#function to check and confirm if the file is already there so we don't download that one
def confirm_file_action(symbol, directory, timeframe="5min", action_name="save"):
    """
    Checks whether a symbol parquet already exists and asks the user what to do.
    Returns:
      True  -> continue with action/save/overwrite
      False -> skip
    """
    directory = Path(directory)
    file_path = directory / f"{symbol}_{timeframe}.parquet"
    if file_path.exists():
        while True:
            answer = input(
                f"{symbol}: file already exists -> {file_path}\n"
                f"Would you like to overwrite it? (y/n): "
            ).strip().lower()
            if answer == "y":
                print(f"{symbol}: overwriting existing file.")
                return True
            if answer == "n":
                print(f"{symbol}: skipping.")
                return False
            print("Please answer with 'y' or 'n'.")
    else:
        while True:
            answer = input(
                f"{symbol}: file does not exist -> {file_path}\n"
                f"Would you like to {action_name} it? (y/n): "
            ).strip().lower()
            if answer == "y":
                print(f"{symbol}: proceeding.")
                return True
            if answer == "n":
                print(f"{symbol}: skipping.")
                return False
            print("Please answer with 'y' or 'n'.")


def Download_5_min_Data(symbols, start, end, timeframe, directory):
    for symbol in symbols:
        if confirm_file_action(symbol, directory, timeframe, action_name="download"):
            if ALPACA_DATA_FEED == "IEX":
                feed = DataFeed.IEX
            else:
                feed = DataFeed.SIP
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=feed,
            )
            bars = client.get_stock_bars(request)
            df = bars.df.reset_index()
            save_symbol_parquets(df, directory, timeframe)

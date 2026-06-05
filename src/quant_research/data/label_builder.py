from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.data.data_downloader import confirm_file_action
from quant_research.utils.paths import PROCESSED_DATA_DIR

Horizon_bars = 12
pct = 0.01

market_open = 6 * 60 + 30
market_close = 13 * 60

expected_regular_session_candles = 78

label_to_ID = {
    "neutral" : 0,
    "up" : 1,
    "down" : 2
}
def load_aligned_data( timeframe = "5min", raw_dir = PROCESSED_DATA_DIR):
    """
    Load raw Alpaca parquet data for one symbol.
    """
    # Makes the path per symbol
    path = Path(raw_dir) / f"aligned_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data for aligned : {path}")
    
    # Reading the data
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("missing timestamp column.")
    
    #changing the timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    #returning the dataframe
    return df
def label_data( timeframe="5min", processed_dir=PROCESSED_DATA_DIR, pct= pct, Horizon_bars=Horizon_bars,):
    #Loading the data
    df = load_aligned_data(
        timeframe=timeframe,
        raw_dir=processed_dir,
    )       
    #making sure it is a datetime
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date
    #this is creating a series that has the time of day since midnight in minutes for each row
    minutes_from_midnight = (
        df["timestamp_pt"].dt.hour * 60
        + df["timestamp_pt"].dt.minute
    )

    #making the mask that will keep the rows with the times in between
    regular_session_mask = (
        (minutes_from_midnight >= market_open)
        & (minutes_from_midnight < market_close)
    )
    df["target_name"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["target_class"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    df["bars_to_barrier"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    # Initially, everything outside regular hours is a warmup row.
    df["label_status"] = "warmup"

    # Regular-session rows initially do not have a complete assigned label.
    df.loc[regular_session_mask, "label_status"] = "incomplete_future_window"
    #this outputs the regular dataframe with the correct times
    rdf = df.loc[regular_session_mask].copy()

    #now that we have this, we need to label it

    #grouping the days
    grouped_days = rdf.groupby("date", sort = False)
    for day, day_df in grouped_days:
        day_df = day_df.sort_values("timestamp")
        if len(day_df) != expected_regular_session_candles:
            raise ValueError(
                f"{day} has {len(day_df)} regular-session candles, "
                f"expected {expected_regular_session_candles}"
            )
        #for each day we check whether the next 12 candles reach the 1% threshold
        for i in range(len(day_df) - Horizon_bars):
            #getting the index to store the label
            original_index = day_df.index[i]
            #getting the open price of the next candle(we are buying at candle i + 1 so we check that when 
            # the next candle opens we will have the 1% change from there)
            open_price = day_df.iloc[i + 1]["NVDA_open"]
            high_barrier = open_price * (1 + pct)
            low_barrier  = open_price * (1 - pct) 
            label = 'neutral'
            for j in range(Horizon_bars):
               curr_high = day_df.iloc[i + 1 + j]["NVDA_high"]
               curr_low = day_df.iloc[i + 1 + j]["NVDA_low"]
               upper_hit = high_barrier <= curr_high
               lower_hit = curr_low <= low_barrier
               if upper_hit and lower_hit:
                    label = "ambiguous"
                    break
               elif upper_hit:
                    label = "up"
                    df.loc[original_index, "bars_to_barrier"] = j + 1
                    break
               elif lower_hit:
                    label = "down"
                    df.loc[original_index, "bars_to_barrier"] = j + 1
                    break
            if label == "ambiguous":
                df.loc[original_index, "label_status"] = "ambiguous_same_bar"
            else:
                df.loc[original_index, "target_name"] = label
                df.loc[original_index, "target_class"] = label_to_ID[label]
                df.loc[original_index, "label_status"] = "valid"
    print(df["target_name"].value_counts(dropna=False))
    print(df["label_status"].value_counts(dropna=False))
    valid_df = df.loc[df["label_status"] == "valid"]
    print(valid_df["target_name"].value_counts(normalize=True))
    labelable_rows = df[
    df["label_status"].isin(["valid", "ambiguous_same_bar"])
    ]

    print(labelable_rows.groupby("date").size().value_counts())
    labelable_counts = labelable_rows.groupby("date").size()

    bad_labelable_days = labelable_counts[
        labelable_counts != expected_regular_session_candles - Horizon_bars
    ]

    if not bad_labelable_days.empty:
        raise ValueError(
            f"Some days have an incorrect number of labelable rows:\n"
            f"{bad_labelable_days.head()}"
        )
    
    ambiguous_count = (
    df["label_status"] == "ambiguous_same_bar"
    ).sum()

    labelable_count = df["label_status"].isin(
        ["valid", "ambiguous_same_bar"]
    ).sum()

    print(f"Ambiguous rate: {ambiguous_count / labelable_count:.4%}")
    barrier_hits = valid_df.loc[
    valid_df["target_name"].isin(["up", "down"])
    ]

    average_bars_to_hit = barrier_hits["bars_to_barrier"].mean()

    print(f"Average bars to hit barrier: {average_bars_to_hit:.2f}")
    print(
        f"Average minutes to hit barrier: "
        f"{average_bars_to_hit * 5:.2f}"
    )
    average_bars_by_direction = (
        barrier_hits
        .groupby("target_name")["bars_to_barrier"]
        .mean()
    )

    print("\nAverage bars to barrier by direction:")
    print(average_bars_by_direction)

    print("\nAverage minutes to barrier by direction:")
    print(average_bars_by_direction * 5)
    output_path = Path(processed_dir) / f"labeled_{timeframe}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved labeled dataset to {output_path}")
    return df







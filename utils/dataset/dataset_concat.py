from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
import tempfile

import pandas as pd
from mlfinpy.data_structure import standard_bars

BarType = Literal[
    "volume",
    "tick",
    "dollar",
    "tick_imbalance",
    "volume_imbalance",
    "dollar_imbalance",
]

def dataset_concat(
    path: str,
    bar_type: BarType,
    original_dataset: pd.DataFrame | None = None,
    threshold: int | None = None,
    output_file_path: str | None = None
):
    """
    Combines multiple CSV files from a given directory into a single DataFrame.

    Each CSV file in `path` is read and converted to `bar_type`, then concatenated with the
    optional `original_dataset`. If `output_file_path` is provided, the result is saved to CSV.

    Parameters
    ----------
    path : str
        Directory containing the CSV files to concatenate.
    bar_type : {"volume", "tick", "dollar", "tick_imbalance", "volume_imbalance", "dollar_imbalance"}
        Bar aggregation type to apply.
    original_dataset : pd.DataFrame, optional
        Existing DataFrame to append to.
    threshold : int, optional
        Positive threshold used by the bar aggregation. For imbalance bars, this is passed
        through to the underlying implementation and typically represents the expected number
        of ticks/observations driving the EMA-based trigger.
    output_file_path : str, optional
        Path to write the concatenated CSV.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame.

    Raises
    ------
    ValueError
        If `threshold` is not a positive integer.
    FileNotFoundError
        If `path` doesn't exist or contains no CSV files.
    NotImplementedError
        If imbalance bars are requested but the installed mlfinpy/standard_bars module
        does not provide the required function.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Specified path doesn't exist: {path}")

    if threshold is None or threshold <= 0:
        raise ValueError(f"Invalid threshold value: {threshold}. Must be a positive integer.")

    folder_path = Path(path)
    csv_files = sorted(folder_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {path}")

    initial_columns = [
        "trade_id",
        "price",
        "qty",
        "quote_qty",
        "time",
        "is_buyer_maker",
        "is_best_match"
    ]

    # Helper to compute bars based on the requested bar_type
    def _compute_bars(df_tick: pd.DataFrame) -> pd.DataFrame:
        # Standard bar families (existing behavior)
        if bar_type == "tick":
            return standard_bars.get_tick_bars(
                df_tick, threshold=threshold, batch_size=1_000_000, verbose=False
            )
        if bar_type == "volume":
            return standard_bars.get_volume_bars(
                df_tick, threshold=threshold, batch_size=1_000_000, verbose=False
            )
        if bar_type == "dollar":
            return standard_bars.get_dollar_bars(
                df_tick, threshold=threshold, batch_size=1_000_000, verbose=False
            )

        # Imbalance bar families (new behavior)
        # We try to call a generic get_imbalance_bars if available (as in common libraries).
        # If your installed version exposes separate helpers, adapt the call sites below.
        imbalance_map = {
            "tick_imbalance": "tick",
            "volume_imbalance": "volume",
            "dollar_imbalance": "dollar",
        }
        if bar_type in imbalance_map:
            if not hasattr(standard_bars, "get_imbalance_bars"):
                # Some versions expose specific functions; try those gracefully.
                fn_name = {
                    "tick_imbalance": "get_tick_imbalance_bars",
                    "volume_imbalance": "get_volume_imbalance_bars",
                    "dollar_imbalance": "get_dollar_imbalance_bars",
                }[bar_type]
                if not hasattr(standard_bars, fn_name):
                    raise NotImplementedError(
                        "Imbalance bars are requested but the installed 'standard_bars' "
                        "module does not provide the required function. "
                        f"Missing: get_imbalance_bars or {fn_name}."
                    )
                # Call the specific imbalance function (signature mirrors the standard bars)
                return getattr(standard_bars, fn_name)(
                    df_tick, threshold=threshold, batch_size=1_000_000, verbose=False
                )

            # Generic imbalance API (most common): pass imbalance type
            # Note: many implementations support an EMA/expected-imbalance window parameter.
            # To preserve the original function's signature, we use a sensible default internally.
            return standard_bars.get_imbalance_bars(
                df_tick,
                threshold=threshold,
                batch_size=1_000_000,
                verbose=False,
                imbalance_type=imbalance_map[bar_type],          # "tick" | "volume" | "dollar"
                expected_imbalance_window=10_000,                 # internal default to avoid API drift
            )

        # Should never get here due to Literal typing, but keep a guard.
        raise ValueError(f"Unsupported bar_type: {bar_type}")

    # Use a temporary file to store intermediate results
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
        temp_filename = temp_file.name
        first_write = True

        # If original_dataset exists, write it first
        if original_dataset is not None:
            original_dataset.to_csv(temp_filename, index=False, mode='w')
            first_write = False

        # Process each CSV file one by one
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df.columns = initial_columns
            df_subset = df[['price','qty','time']]

            df_tick = df_subset.rename(columns={"qty": "volume", "time": "date_time"})
            df_tick["date_time"] = pd.to_datetime(df_tick["date_time"], unit="us", utc=True)
            df_tick = df_tick.reindex(columns=["date_time", "price", "volume"])

            bars = _compute_bars(df_tick)

            # Write to temporary file
            bars.to_csv(temp_filename, index=False, mode='a', header=first_write)
            first_write = False

            # Clear memory
            del df, df_subset, df_tick, bars

    # Now sort the temporary file by date_time and remove duplicates
    # Process in chunks to avoid loading everything into memory
    chunk_size = 100_000
    sorted_chunks = []

    # Read and sort in chunks
    for chunk in pd.read_csv(temp_filename, chunksize=chunk_size):
        if "date_time" in chunk.columns:
            chunk["date_time"] = pd.to_datetime(chunk["date_time"])
            sorted_chunks.append(chunk.sort_values("date_time"))
        else:
            sorted_chunks.append(chunk)

    # Merge sorted chunks
    result = pd.concat(sorted_chunks, ignore_index=True)

    # Final sort and deduplication
    if "date_time" in result.columns:
        result = result.sort_values("date_time").drop_duplicates().reset_index(drop=True)

    # Clean up temporary file
    os.unlink(temp_filename)

    # Save to output file if specified
    if output_file_path:
        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file_path, index=False)

    return result

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import openmeteo_requests
import pandas as pd

# Variables requested from Open-Meteo; keep in one place for reuse.
REQUIRED_VARS = (
    "temperature_2m",
    "rain",
    "snowfall",
    "relative_humidity_2m",
    "wind_speed_10m",
    "weather_code",
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _build_client() -> openmeteo_requests.Client:
    # Single attempt client; no retries/caching. One script run = one try.
    return openmeteo_requests.Client()


def fetch_weather_batches(
    *,
    start_date: str,
    end_date: str,
    chunk_start: int = 0,
    chunk_size: int = 100,
    timezone: str = "Europe/Brussels",
    nodes_path: Path = Path("data/silver/static/op_nodes.parquet"),
    raw_dir: Path = Path("data/raw/weather"),
) -> Path:
    """
    Fetch exactly one batch of weather data starting at `chunk_start`.

    The batch pulls up to `chunk_size` operational points from the op_nodes parquet
    (which is built in the silver layer) and writes one parquet file to `raw_dir`.

    The intent is to call this script manually/with a scheduler at the desired cadence
    to respect the Open-Meteo free tier limits.

    Observed limits for requests of 100 nodes and 3 years:
    - 1 per minute.
    - 2 per hour.
    - 3 per day.
    """
    nodes = pd.read_parquet(nodes_path, columns=["op_id", "lat", "lon"])
    coords = list(zip(nodes["lat"].tolist(), nodes["lon"].tolist()))
    op_ids = nodes["op_id"].tolist()
    total = len(coords)

    client = _build_client()

    if chunk_start >= total:
        raise ValueError(f"No batches to fetch: chunk_start={chunk_start} is beyond available nodes ({total}).")

    frames = []
    end = min(chunk_start + chunk_size, total)
    lat_batch, lon_batch = zip(*coords[chunk_start:end])
    op_id_batch = op_ids[chunk_start:end]

    params = {
        "latitude": list(lat_batch),
        "longitude": list(lon_batch),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": list(REQUIRED_VARS),
        "timezone": timezone,
    }

    responses = client.weather_api(ARCHIVE_URL, params=params)
    if len(responses) != (end - chunk_start):
        raise RuntimeError(
            f"Expected {end - chunk_start} responses for batch {chunk_start}-{end-1}, got {len(responses)}."
        )

    for idx, resp in enumerate(responses):
        hourly = resp.Hourly()
        timestamps = np.arange(hourly.Time(), hourly.TimeEnd(), hourly.Interval())
        time_index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone)
        record: dict[str, Any] = {
            "op_id": op_id_batch[idx],
            "time": time_index,
        }
        for var_idx, var_name in enumerate(REQUIRED_VARS):
            record[var_name] = hourly.Variables(var_idx).ValuesAsNumpy()
        frames.append(pd.DataFrame(record))

    if not frames:
        raise RuntimeError("No weather data frames were produced; aborting.")

    data = pd.concat(frames, ignore_index=True)

    chunk_end = end - 1
    output_path = raw_dir / f"weather_{chunk_start}-{chunk_end}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)

    print(
        f"[weather_fetch] Saved {len(data):,} rows for batches "
        f"{chunk_start}-{chunk_end} to {output_path}"
    )
    return output_path


def concat_weather_raw(
    *,
    sources: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Concatenate raw weather parquet files (expect the manifest to eager-load them).
    manifest_runner._load_source_dataframe already handles the concatenation, so just return a copy here. 
    """
    df = sources["main"].copy()
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    df = df.sort_values(["op_id", "time"])

    weather_cols = [c for c in REQUIRED_VARS if c in df.columns]

    def _reindex_and_ffill(g: pd.DataFrame) -> pd.DataFrame:
        op_id = g.name
        g = g.sort_values("time").drop_duplicates(subset=["time"], keep="last")
        g = g.set_index("time")
        g = g.asfreq("h")
        g["op_id"] = op_id
        g[weather_cols] = g[weather_cols].ffill()
        return g.reset_index()

    df = df.groupby("op_id", group_keys=False).apply(_reindex_and_ffill).reset_index(drop=True)
    return df


__all__ = ["fetch_weather_batches", "concat_weather_raw", "REQUIRED_VARS"]

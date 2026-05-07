from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset.pipeline.helpers import write_yaml
from src.dataset.gold.helpers import (
    TAIL_SAFETY_BUFFER_MIN,
    build_gold_eval_table,
    sample_train_test_snapshots,
)


def _month_range(start_day: str, end_day: str) -> list[str]:
    start = pd.Timestamp(start_day).to_period("M")
    end = pd.Timestamp(end_day).to_period("M")
    return [p.strftime("%Y%m") for p in pd.period_range(start=start, end=end, freq="M")]


def build_core_dataset(
    *,
    start_train_day: str,
    end_train_day: str,
    start_test_day: str,
    end_test_day: str,
    n_train: int,
    n_test: int,
    n_future: int,
    idle_time_beg: int,
    idle_time_end: int,
    output_root: Path,
    seed: int = 42,
    events_dir: Path = Path("data/silver/events"),
    journeys_dir: Path = Path("data/silver/journeys"),
    missing_event_placeholder: int = -1,
    build_train_eval_table: bool = False,
    show_progress: bool = True,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_core_spec_output = output_root / "dataset_core_spec.yaml"
    test_eval_output = output_root / "test_eval_table.parquet"
    train_eval_output = output_root / "train_eval_table.parquet"
    metadata_output = output_root / "metadata.yaml"

    months = sorted(
        set(
            _month_range(start_train_day, end_train_day)
            + _month_range(start_test_day, end_test_day)
        )
    )
    journeys = pd.concat(
        [
                pd.read_parquet(
                    journeys_dir / f"journeys_{month}.parquet",
                    columns=["service_date", "start_observed_ts", "end_observed_ts", "start_planned_ts"],
                )
            for month in months
        ],
        ignore_index=True,
    )

    splits = sample_train_test_snapshots(
        start_train_day=start_train_day,
        end_train_day=end_train_day,
        start_test_day=start_test_day,
        end_test_day=end_test_day,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        journeys=journeys,
        idle_time_beg=idle_time_beg,
        idle_time_end=idle_time_end,
    )
    payload = {
        "start_train_day": start_train_day,
        "end_train_day": end_train_day,
        "start_test_day": start_test_day,
        "end_test_day": end_test_day,
        "n_train": n_train,
        "n_test": n_test,
        "n_future": n_future,
        "idle_time_beg": idle_time_beg,
        "idle_time_end": idle_time_end,
        "tail_safety_buffer_min": TAIL_SAFETY_BUFFER_MIN,
        "seed": seed,
        **splits,
    }
    write_yaml(dataset_core_spec_output, payload)
    print(
        f"[gold_core] Wrote dataset core spec to {dataset_core_spec_output} "
        f"(train={len(splits['train_snapshots'])}, test={len(splits['test_snapshots'])})"
    )

    gold_eval = build_gold_eval_table(
        events_dir=events_dir,
        journeys_dir=journeys_dir,
        snapshot_config=payload,
        missing_event_placeholder=missing_event_placeholder,
        build_train_eval_table=build_train_eval_table,
        show_progress=show_progress,
    )
    gold_eval["test_eval_table"].to_parquet(test_eval_output, index=False)
    if build_train_eval_table:
        gold_eval["train_eval_table"].to_parquet(train_eval_output, index=False)
    print(f"[gold_core] Wrote gold eval tables to {output_root}")

    metadata = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "inputs": {
            "events_dir": str(events_dir),
            "journeys_dir": str(journeys_dir),
        },
        "parameters": {
            "start_train_day": start_train_day,
            "end_train_day": end_train_day,
            "start_test_day": start_test_day,
            "end_test_day": end_test_day,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_future": int(n_future),
            "idle_time_beg": int(idle_time_beg),
            "idle_time_end": int(idle_time_end),
            "tail_safety_buffer_min": int(TAIL_SAFETY_BUFFER_MIN),
            "missing_event_placeholder": int(missing_event_placeholder),
            "seed": int(seed),
            "build_train_eval_table": bool(build_train_eval_table),
        },
        "outputs": {
            "dataset_core_spec_path": str(dataset_core_spec_output),
            "test_eval_table_path": str(test_eval_output),
            "train_eval_table_path": str(train_eval_output) if build_train_eval_table else None,
        },
        "counts": {
            "n_train_snapshots": int(len(splits["train_snapshots"])),
            "n_test_snapshots": int(len(splits["test_snapshots"])),
            "n_test_eval_rows": int(len(gold_eval["test_eval_table"])),
            "n_train_eval_rows": int(len(gold_eval["train_eval_table"])) if build_train_eval_table else 0,
        },
    }
    write_yaml(metadata_output, metadata)
    print(f"[gold_core] Wrote metadata to {metadata_output}")

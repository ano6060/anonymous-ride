from __future__ import annotations

import gc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from src.dataset.pipeline.manifest_runner import apply_manifest


def _strip_bom(text: str) -> str:
    return text.lstrip("\ufeff") if text.startswith("\ufeff") else text


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {(_strip_bom(k) if isinstance(k, str) else k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    if isinstance(obj, str):
        return _strip_bom(obj)
    return obj


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8")
    return _sanitize(yaml.safe_load(text) or {})


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8")
    return _sanitize(yaml.safe_load(text) or {})


def manifest_paths(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(root.glob("*.yaml"))
    raise FileNotFoundError(f"No manifest files found under {root}")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def resolve_output_path(io_cfg: Dict[str, Any], wildcards: Optional[Dict[str, str]] = None) -> Path:
    output_cfg = io_cfg.get("output") or {}
    template = output_cfg.get("path")
    if not template:
        raise ValueError("Manifest io.output must include a path.")
    path = template
    if wildcards:
        token = wildcards.get("output") or wildcards.get("main")
        if token:
            path = path.replace("*", token)
    return Path(path)


def execute_manifest(
    *,
    manifest: Dict[str, Any],
    manifest_path: Path,
    metadata: Dict[str, Any],
    wildcards: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    df = apply_manifest(manifest, wildcards=wildcards)
    output_path = resolve_output_path(manifest["io"], wildcards)
    entry: Dict[str, Any] = {
        "dataset": manifest.get("id"),
        "manifest_path": str(manifest_path),
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "path": str(output_path),
        "rows": len(df),
    }
    if wildcards:
        entry["wildcards"] = dict(wildcards)
    metadata.setdefault("datasets", []).append(entry)
    return df


__all__ = [
    "read_yaml",
    "load_manifest",
    "manifest_paths",
    "write_yaml",
    "free",
    "resolve_output_path",
    "execute_manifest",
]

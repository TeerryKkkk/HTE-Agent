from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise ValueError(f"Unable to decode CSV at {path}") from last_error


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _normalize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(_normalize_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    normalized_rows = [{key: _normalize_value(value) for key, value in row.items()} for row in rows]
    if not normalized_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in normalized_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def write_markdown(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def slugify(text: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "item"


def bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

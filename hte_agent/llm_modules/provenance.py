"""Compact call provenance; excludes prompts, completions, headers and credentials."""
from __future__ import annotations

import json

from hte_agent.shared.candidate_schema import model_response_succeeded


def call_record(result, *, round_index: int) -> dict[str, object]:
    raw = result.raw_response
    return {
        "round_index": round_index,
        "invoked": result.invoked,
        "succeeded": model_response_succeeded(result),
        "status": result.status,
        "model": result.model,
        "response_model": raw.get("response_model", ""),
        "attempts": [{key: attempt.get(key, "") for key in ("model", "response_model", "status")}
                     for attempt in raw.get("attempts", [])],
    }


def summarize_calls(prefix: str, calls: list[dict[str, object]], *, enabled: bool,
                    expected_count: int) -> dict[str, object]:
    statuses = sorted({str(call["status"]) for call in calls})
    models = sorted({str(call["model"]) for call in calls if call["model"]})
    succeeded = bool(expected_count) and len(calls) == expected_count and all(call["succeeded"] for call in calls)
    status = ("disabled" if not enabled else "not_run") if not calls else statuses[0] if len(statuses) == 1 else "mixed"
    return {
        f"{prefix}_enabled": enabled,
        f"{prefix}_invoked": any(call["invoked"] for call in calls),
        f"{prefix}_succeeded": succeeded,
        f"{prefix}_status": status,
        f"{prefix}_model": models[0] if len(models) == 1 else "",
        f"{prefix}_models": json.dumps(models),
        f"{prefix}_call_count": len(calls),
        f"{prefix}_successful_call_count": sum(bool(call["succeeded"]) for call in calls),
        f"{prefix}_calls": json.dumps(calls, sort_keys=True),
    }

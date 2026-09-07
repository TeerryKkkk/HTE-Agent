from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import isfinite
from typing import Any

from hte_agent.config import SharedRunConstants
from hte_agent.shared.candidate_schema import Candidate, LLMS1Critique, model_response_succeeded
from hte_agent.shared.io_utils import bounded
from hte_agent.shared.task_representation import TaskRepresentation
from hte_agent.stage1.retrieval_initializer import RetrievedSupportBundle

_POSITIVE_REASON = "positive_evidence"
_DEFAULT_NEGATIVE_REASON_BY_AXIS = {
    "ligand_label": "weak_transfer",
    "base_label": "dubious_compatibility",
    "solvent_label": "dubious_compatibility",
}
_REASON_PRIORITY = {
    "conflict": 4,
    "dubious_compatibility": 3,
    "weak_transfer": 2,
    "low_support": 1,
    "positive_evidence": 1,
    "other": 1,
}


def _normalize_reason_label(label: object, axis: str, *, positive: bool, combination: bool = False) -> str:
    candidate = str(label or "").strip().lower()
    if candidate in {"conflict", "weak_transfer", "dubious_compatibility", "low_support", "positive_evidence"}:
        return candidate
    if positive:
        return _POSITIVE_REASON
    if combination:
        return "conflict"
    return _DEFAULT_NEGATIVE_REASON_BY_AXIS.get(axis, "other")


def _parse_label_map(raw_payload: object) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    if not isinstance(raw_payload, dict):
        return parsed
    for axis, value_map in raw_payload.items():
        if not isinstance(value_map, dict):
            continue
        parsed[str(axis)] = {str(value): str(label) for value, label in value_map.items()}
    return parsed


def _parse_value_map(raw_payload: object, upper_bound: float) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    if not isinstance(raw_payload, dict):
        return parsed
    for axis, value_map in raw_payload.items():
        if not isinstance(value_map, dict):
            continue
        normalized_axis: dict[str, float] = {}
        for value, magnitude in value_map.items():
            try:
                numeric = float(magnitude)
                if isinstance(magnitude, bool) or not isfinite(numeric):
                    continue
                normalized_magnitude = bounded(numeric, 0.0, upper_bound)
            except (TypeError, ValueError):
                continue
            normalized_axis[str(value)] = normalized_magnitude
        if normalized_axis:
            parsed[str(axis)] = normalized_axis
    return parsed


def _entry_priority(entry: dict[str, Any]) -> tuple[float, float, float, str]:
    coverage_fraction = max(float(entry["coverage_fraction"]), 0.01)
    reason_priority = _REASON_PRIORITY.get(str(entry["reason_label"]), 1)
    action_multiplier = 1.0 if str(entry["action"]) == "penalty" else 0.75
    return (
        round(reason_priority * action_multiplier * float(entry["magnitude"]) / coverage_fraction, 6),
        -coverage_fraction,
        float(entry["magnitude"]),
        str(entry["entry_id"]),
    )


def _candidate_ids_for_value(candidates: list[Candidate], axis: str, value: str) -> set[str]:
    return {candidate.candidate_id for candidate in candidates if str(candidate.parameters.get(axis, "")) == str(value)}


def _candidate_ids_for_combination(candidates: list[Candidate], combination: dict[str, str]) -> set[str]:
    return {
        candidate.candidate_id
        for candidate in candidates
        if all(str(candidate.parameters.get(axis, "")) == str(value) for axis, value in combination.items())
    }


def _build_diagnostics(
    candidate_count: int,
    contradiction_resolution_count: int,
    reject_downgrade_count: int,
    dropped_for_coverage_count: int,
    dropped_for_budget_count: int,
    kept_bonus_entries: list[dict[str, Any]],
    kept_penalty_entries: list[dict[str, Any]],
    kept_reject_combination_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    estimated_modified_ids: set[str] = set()
    estimated_reject_ids: set[str] = set()
    reason_counts: Counter[str] = Counter()
    for entry in [*kept_bonus_entries, *kept_penalty_entries, *kept_reject_combination_entries]:
        estimated_modified_ids.update(entry["candidate_ids"])
        reason_counts[str(entry["reason_label"])] += len(entry["candidate_ids"])
        if str(entry["action"]) == "reject":
            estimated_reject_ids.update(entry["candidate_ids"])
    return {
        "candidate_pool_size_before_s1": candidate_count,
        "candidate_pool_size_after_s1": candidate_count,
        "estimated_modified_candidate_count": len(estimated_modified_ids),
        "estimated_modified_fraction": round(len(estimated_modified_ids) / candidate_count, 4) if candidate_count else 0.0,
        "estimated_reject_candidate_count": len(estimated_reject_ids),
        "contradiction_resolution_count": contradiction_resolution_count,
        "reject_downgrade_count": reject_downgrade_count,
        "dropped_for_coverage_count": dropped_for_coverage_count,
        "dropped_for_budget_count": dropped_for_budget_count,
        "final_action_entry_counts": {
            "bonus": len(kept_bonus_entries),
            "penalty": len(kept_penalty_entries),
            "reject": len(kept_reject_combination_entries),
        },
        "final_reason_label_counts": dict(reason_counts),
    }


def calibrate_precedent_critique(
    critique: LLMS1Critique,
    candidate_universe: list[Candidate],
    constants: SharedRunConstants,
) -> LLMS1Critique:
    if not model_response_succeeded(critique):
        return replace(critique, bonuses={}, penalties={}, reject_values={}, reject_combinations=[],
                       bonus_reason_labels={}, penalty_reason_labels={}, reject_value_reason_labels={},
                       reject_combination_reason_labels=[], diagnostics={},
                       notes=list(critique.notes) + ["no_model_actions_applied"])
    candidate_count = len(candidate_universe)
    max_modified_count = max(0, int(candidate_count * constants.s1_max_modified_fraction))
    max_reject_count = max(0, int(candidate_count * constants.s1_max_reject_fraction))

    contradiction_resolution_count = 0
    reject_downgrade_count = 0
    dropped_for_coverage_count = 0
    dropped_for_budget_count = 0

    raw_bonus_entries: dict[tuple[str, str], dict[str, Any]] = {}
    raw_penalty_entries: dict[tuple[str, str], dict[str, Any]] = {}
    raw_reject_value_entries: dict[tuple[str, str], dict[str, Any]] = {}

    for axis, value_map in critique.bonuses.items():
        for value, magnitude in value_map.items():
            raw_bonus_entries[(axis, str(value))] = {
                "entry_id": f"bonus:{axis}={value}",
                "action": "bonus",
                "axis": axis,
                "value": str(value),
                "magnitude": bounded(float(magnitude), 0.0, constants.s1_max_bonus_magnitude),
                "reason_label": _normalize_reason_label(
                    critique.bonus_reason_labels.get(axis, {}).get(str(value), ""),
                    axis,
                    positive=True,
                ),
            }
    for axis, value_map in critique.penalties.items():
        for value, magnitude in value_map.items():
            raw_penalty_entries[(axis, str(value))] = {
                "entry_id": f"penalty:{axis}={value}",
                "action": "penalty",
                "axis": axis,
                "value": str(value),
                "magnitude": bounded(float(magnitude), 0.0, constants.s1_max_penalty_magnitude),
                "reason_label": _normalize_reason_label(
                    critique.penalty_reason_labels.get(axis, {}).get(str(value), ""),
                    axis,
                    positive=False,
                ),
            }
    for axis, values in critique.reject_values.items():
        for value in values:
            raw_reject_value_entries[(axis, str(value))] = {
                "entry_id": f"reject_value:{axis}={value}",
                "action": "penalty",
                "axis": axis,
                "value": str(value),
                "magnitude": constants.s1_max_penalty_magnitude,
                "reason_label": _normalize_reason_label(
                    critique.reject_value_reason_labels.get(axis, {}).get(str(value), ""),
                    axis,
                    positive=False,
                ),
            }
            reject_downgrade_count += 1

    resolved_penalty_entries: list[dict[str, Any]] = []
    kept_bonus_candidates: list[dict[str, Any]] = []
    all_scope_keys = set(raw_bonus_entries) | set(raw_penalty_entries) | set(raw_reject_value_entries)
    for axis, value in sorted(all_scope_keys):
        bonus_entry = raw_bonus_entries.get((axis, value))
        negative_candidates = [
            entry
            for entry in (raw_penalty_entries.get((axis, value)), raw_reject_value_entries.get((axis, value)))
            if entry is not None
        ]
        if bonus_entry and negative_candidates:
            contradiction_resolution_count += 1
            strongest_negative = max(
                negative_candidates,
                key=lambda entry: (_REASON_PRIORITY.get(str(entry["reason_label"]), 1), float(entry["magnitude"])),
            )
            net_negative = float(strongest_negative["magnitude"]) - float(bonus_entry["magnitude"])
            if net_negative >= constants.s1_min_effect_to_keep:
                resolved_penalty_entries.append(
                    {
                        "entry_id": f"resolved_penalty:{axis}={value}",
                        "action": "penalty",
                        "axis": axis,
                        "value": value,
                        "magnitude": bounded(net_negative, 0.0, constants.s1_contradiction_penalty_magnitude),
                        "reason_label": str(strongest_negative["reason_label"]),
                    }
                )
            continue
        if bonus_entry:
            kept_bonus_candidates.append(bonus_entry)
            continue
        if negative_candidates:
            strongest_negative = max(
                negative_candidates,
                key=lambda entry: (_REASON_PRIORITY.get(str(entry["reason_label"]), 1), float(entry["magnitude"])),
            )
            resolved_penalty_entries.append(strongest_negative)

    candidate_sets: dict[str, set[str]] = {}
    for entry in [*kept_bonus_candidates, *resolved_penalty_entries]:
        ids = _candidate_ids_for_value(candidate_universe, str(entry["axis"]), str(entry["value"]))
        if not ids:
            continue
        coverage_fraction = len(ids) / candidate_count if candidate_count else 0.0
        if coverage_fraction > constants.s1_max_single_entry_fraction:
            dropped_for_coverage_count += 1
            continue
        entry["candidate_ids"] = ids
        entry["coverage_fraction"] = coverage_fraction
        candidate_sets[entry["entry_id"]] = ids

    reject_entries: list[dict[str, Any]] = []
    for index, combination in enumerate(critique.reject_combinations):
        if not combination:
            continue
        reason_label = _normalize_reason_label(
            critique.reject_combination_reason_labels[index] if index < len(critique.reject_combination_reason_labels) else "",
            "",
            positive=False,
            combination=True,
        )
        if any(
            (axis, value) in raw_bonus_entries
            for axis, value in combination.items()
        ):
            contradiction_resolution_count += 1
            continue
        ids = _candidate_ids_for_combination(candidate_universe, combination)
        if not ids:
            continue
        coverage_fraction = len(ids) / candidate_count if candidate_count else 0.0
        if coverage_fraction > constants.s1_max_single_entry_fraction:
            dropped_for_coverage_count += 1
            continue
        reject_entries.append(
            {
                "entry_id": f"reject_combination:{index}",
                "action": "reject",
                "combination": dict(combination),
                "magnitude": constants.s1_max_penalty_magnitude,
                "reason_label": reason_label,
                "candidate_ids": ids,
                "coverage_fraction": coverage_fraction,
            }
        )

    modified_ids: set[str] = set()
    reject_ids: set[str] = set()
    kept_reject_entries: list[dict[str, Any]] = []
    for entry in sorted(reject_entries, key=_entry_priority, reverse=True):
        if len(kept_reject_entries) >= constants.s1_max_reject_combination_entries:
            dropped_for_budget_count += 1
            continue
        if len(reject_ids | entry["candidate_ids"]) > max_reject_count:
            dropped_for_budget_count += 1
            continue
        if len(modified_ids | entry["candidate_ids"]) > max_modified_count:
            dropped_for_budget_count += 1
            continue
        kept_reject_entries.append(entry)
        reject_ids.update(entry["candidate_ids"])
        modified_ids.update(entry["candidate_ids"])

    kept_penalty_entries: list[dict[str, Any]] = []
    for entry in sorted(
        [entry for entry in resolved_penalty_entries if "candidate_ids" in entry and entry["magnitude"] >= constants.s1_min_effect_to_keep],
        key=_entry_priority,
        reverse=True,
    ):
        if len(kept_penalty_entries) >= constants.s1_max_penalty_entries:
            dropped_for_budget_count += 1
            continue
        if len(modified_ids | entry["candidate_ids"]) > max_modified_count:
            dropped_for_budget_count += 1
            continue
        kept_penalty_entries.append(entry)
        modified_ids.update(entry["candidate_ids"])

    kept_bonus_entries: list[dict[str, Any]] = []
    for entry in sorted(
        [entry for entry in kept_bonus_candidates if "candidate_ids" in entry and entry["magnitude"] >= constants.s1_min_effect_to_keep],
        key=_entry_priority,
        reverse=True,
    ):
        if len(kept_bonus_entries) >= constants.s1_max_bonus_entries:
            dropped_for_budget_count += 1
            continue
        if len(modified_ids | entry["candidate_ids"]) > max_modified_count:
            dropped_for_budget_count += 1
            continue
        kept_bonus_entries.append(entry)
        modified_ids.update(entry["candidate_ids"])

    bonuses: dict[str, dict[str, float]] = {}
    bonus_reason_labels: dict[str, dict[str, str]] = {}
    for entry in kept_bonus_entries:
        bonuses.setdefault(str(entry["axis"]), {})[str(entry["value"])] = round(float(entry["magnitude"]), 4)
        bonus_reason_labels.setdefault(str(entry["axis"]), {})[str(entry["value"])] = str(entry["reason_label"])

    penalties: dict[str, dict[str, float]] = {}
    penalty_reason_labels: dict[str, dict[str, str]] = {}
    for entry in kept_penalty_entries:
        penalties.setdefault(str(entry["axis"]), {})[str(entry["value"])] = round(float(entry["magnitude"]), 4)
        penalty_reason_labels.setdefault(str(entry["axis"]), {})[str(entry["value"])] = str(entry["reason_label"])

    reject_combinations = [dict(entry["combination"]) for entry in kept_reject_entries]
    reject_combination_reason_labels = [str(entry["reason_label"]) for entry in kept_reject_entries]

    diagnostics = _build_diagnostics(
        candidate_count,
        contradiction_resolution_count,
        reject_downgrade_count,
        dropped_for_coverage_count,
        dropped_for_budget_count,
        kept_bonus_entries,
        kept_penalty_entries,
        kept_reject_entries,
    )
    diagnostics.update(
        {
            "max_candidate_bonus_magnitude": constants.s1_max_bonus_magnitude,
            "max_candidate_penalty_magnitude": constants.s1_max_penalty_magnitude,
            "max_modified_fraction": constants.s1_max_modified_fraction,
            "max_reject_fraction": constants.s1_max_reject_fraction,
            "max_single_entry_fraction": constants.s1_max_single_entry_fraction,
        }
    )

    return LLMS1Critique(
        invoked=critique.invoked,
        status="success_calibrated",
        model=critique.model,
        summary=critique.summary,
        bonuses=bonuses,
        penalties=penalties,
        reject_values={},
        reject_combinations=reject_combinations,
        bonus_reason_labels=bonus_reason_labels,
        penalty_reason_labels=penalty_reason_labels,
        reject_value_reason_labels={},
        reject_combination_reason_labels=reject_combination_reason_labels,
        notes=list(critique.notes) + ["s1_calibration_applied"],
        diagnostics=diagnostics,
        raw_response=critique.raw_response,
    )


def _parse_model_critique(response: dict[str, object]) -> LLMS1Critique:
    parsed = response.get("parsed", {})
    if not isinstance(parsed, dict) or any(
        key in parsed and not isinstance(parsed[key], kind)
        for key, kind in (("bonuses", dict), ("penalties", dict), ("reject_values", dict),
                          ("reject_combinations", list), ("notes", list),
                          ("reject_combination_reason_labels", list))
    ):
        return LLMS1Critique(bool(response["invoked"]), "invalid_response", str(response["model"]),
                             "Invalid critique schema; no correction applied.", raw_response=response)
    combinations = []
    combination_labels = []
    labels = parsed.get("reject_combination_reason_labels", [])
    for index, item in enumerate(parsed.get("reject_combinations", [])):
        if isinstance(item, dict) and item:
            combinations.append({str(key): str(value) for key, value in item.items()})
            combination_labels.append(str(labels[index]) if index < len(labels) else "conflict")
    return LLMS1Critique(
        invoked=bool(response["invoked"]), status="success", model=str(response["model"]),
        summary=str(parsed.get("summary", "")),
        bonuses=_parse_value_map(parsed.get("bonuses"), 0.15),
        penalties=_parse_value_map(parsed.get("penalties"), 0.15),
        reject_values={str(axis): [str(value) for value in values]
                       for axis, values in parsed.get("reject_values", {}).items() if isinstance(values, list)},
        reject_combinations=combinations,
        bonus_reason_labels=_parse_label_map(parsed.get("bonus_reason_labels")),
        penalty_reason_labels=_parse_label_map(parsed.get("penalty_reason_labels")),
        reject_value_reason_labels=_parse_label_map(parsed.get("reject_value_reason_labels")),
        reject_combination_reason_labels=combination_labels,
        notes=[str(item) for item in parsed.get("notes", [])], raw_response=response,
    )


def run_precedent_critic(
    task: TaskRepresentation,
    support: RetrievedSupportBundle,
    client,
    constants: SharedRunConstants,
) -> LLMS1Critique:
    payload = {
        "task_profile": task.profile_name,
        "substrate_features": list(task.substrate_features),
        "constraints": list(task.constraints),
        "guidance": {
            "default_action": "keep_or_no_change",
            "prefer_penalty_over_reject": True,
            "only_reject_for_high_confidence_conflict": True,
            "max_bonus": constants.s1_max_bonus_magnitude,
            "max_penalty": constants.s1_max_penalty_magnitude,
        },
        "precedent_pool": [
            {
                "record_id": record.record_id,
                "ligand_label": record.ligand_label,
                "base_label": record.base_label,
                "solvent_label": record.solvent_label,
                "outcome_label": record.outcome_label,
                "outcome_yield": record.outcome_yield,
                "retrieval_score": record.retrieval_score,
            }
            for record in support.precedents[:10]
        ],
        "failure_support": [
            {
                "record_id": record.record_id,
                "ligand_label": record.ligand_label,
                "base_label": record.base_label,
                "solvent_label": record.solvent_label,
                "retrieval_score": record.retrieval_score,
            }
            for record in support.failure_cases[:6]
        ],
        "output_schema": {
            "summary": "short string",
            "bonuses": {"ligand_label": {"value": "0.00-0.03"}},
            "penalties": {"base_label": {"value": "0.00-0.05"}},
            "reject_values": {"base_label": ["value"]},
            "bonus_reason_labels": {"ligand_label": {"value": "positive_evidence"}},
            "penalty_reason_labels": {"base_label": {"value": "conflict|dubious_compatibility|weak_transfer|low_support"}},
            "reject_value_reason_labels": {"base_label": {"value": "conflict|dubious_compatibility"}},
            "reject_combination_reason_labels": ["conflict"],
            "reject_combinations": [{"ligand_label": "value", "base_label": "value", "solvent_label": "value"}],
            "notes": ["short reason"],
        },
    }
    response = client.complete_json(
        "precedent_critic",
        (
            "You are the LLM-S1 precedent critic for a chemistry optimization pipeline. "
            "Return JSON only. Default to keep/no change. Prefer bounded penalties over rejects. "
            "Only suggest reject for high-confidence, strongly supported incompatibilities. "
            "Do not generate experiments."
        ),
        payload,
    )
    if response["status"] != "success" or not response["invoked"]:
        return LLMS1Critique(
            invoked=bool(response["invoked"]), status=str(response["status"]), model=str(response["model"]),
            summary="No successful model response; no correction applied.", raw_response=response,
        )
    return _parse_model_critique(response)

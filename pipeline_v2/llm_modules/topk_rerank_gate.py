from __future__ import annotations

from collections import Counter
import re
from typing import Any

from pipeline_v2.config import SharedRunConstants
from pipeline_v2.shared.candidate_schema import Candidate, LLMS2Gate, LLMS2Summary, RoundAggregate, Stage2Score
from pipeline_v2.shared.io_utils import bounded, mean
from pipeline_v2.shared.task_representation import TaskRepresentation


def _decision_priority(decision: str) -> int:
    return {"reject": 4, "downweight": 3, "prefer_for_exploration": 2, "keep": 1}.get(decision, 0)


def _profile_limits(task: TaskRepresentation, constants: SharedRunConstants) -> dict[str, float | int | bool]:
    is_hindered = task.profile_name == "hindered_aryl_chloride_activation"
    return {
        "is_hindered": is_hindered,
        "min_nonkeep_confidence": constants.s2_hindered_min_confidence_for_nonkeep if is_hindered else constants.s2_min_confidence_for_nonkeep,
        "min_reject_confidence": constants.s2_hindered_min_confidence_for_reject if is_hindered else constants.s2_min_confidence_for_reject,
        "max_nonkeep_actions": min(constants.s2_max_nonkeep_actions, 2) if is_hindered else constants.s2_max_nonkeep_actions,
        "max_downweight_actions": min(constants.s2_max_downweight_actions, 1) if is_hindered else constants.s2_max_downweight_actions,
        "max_prefer_actions": 0 if is_hindered else constants.s2_max_prefer_actions,
        "max_reject_actions": 0 if is_hindered else constants.s2_max_reject_actions,
        "max_final_batch_changes": constants.s2_hindered_max_final_batch_changes if is_hindered else constants.s2_max_final_batch_changes,
        "boundary_band": constants.s2_hindered_boundary_band if is_hindered else constants.s2_boundary_band,
    }


def _seed_keep_actions(topk_candidates: list[Candidate], score_context: dict[str, dict[str, float | int | bool]]) -> dict[str, dict[str, Any]]:
    return {
        candidate.candidate_id: {
            "candidate_id": candidate.candidate_id,
            "decision": "keep",
            "score_delta": 0.0,
            "reason": "default_keep",
            "confidence": 1.0,
            "rank": int(score_context[candidate.candidate_id]["rank"]),
        }
        for candidate in topk_candidates
    }


def _build_score_context(topk_scores: list[Stage2Score], batch_size: int, boundary_band: int) -> dict[str, dict[str, float | int | bool]]:
    if not topk_scores:
        return {}
    cutoff_index = min(max(batch_size - 1, 0), len(topk_scores) - 1)
    cutoff_score = float(topk_scores[cutoff_index].acquisition_score)
    cutoff_mean = float(topk_scores[cutoff_index].predicted_mean)
    context: dict[str, dict[str, float | int | bool]] = {}
    for rank, score in enumerate(topk_scores, start=1):
        context[score.candidate_id] = {
            "rank": rank,
            "inside_batch": rank <= batch_size,
            "near_boundary": abs(rank - batch_size) <= boundary_band,
            "batch_distance": abs(rank - batch_size),
            "acquisition_score": float(score.acquisition_score),
            "predicted_mean": float(score.predicted_mean),
            "predicted_std": float(score.predicted_std),
            "score_gap_to_cutoff": round(float(score.acquisition_score) - cutoff_score, 4),
            "predicted_mean_gap_to_cutoff": round(float(score.predicted_mean) - cutoff_mean, 4),
        }
    return context


def _estimate_confidence(item: dict[str, Any], decision: str, constants: SharedRunConstants) -> float:
    if item.get("confidence") not in (None, ""):
        return bounded(_coerce_float(item.get("confidence", 0.0), 0.0), 0.0, 1.0)
    if decision == "keep":
        return 1.0
    max_delta = max(abs(constants.s2_downweight_delta), abs(constants.s2_reject_delta), abs(constants.s2_prefer_delta), 1e-6)
    delta_term = min(abs(float(item.get("score_delta", 0.0))) / max_delta, 1.0)
    reason = str(item.get("reason", "")).lower()
    heuristic_bonus = 0.12 if "underperform" in reason or "consistent" in reason else 0.0
    if decision == "reject":
        base = 0.74
    elif decision == "downweight":
        base = 0.70
    else:
        base = 0.68
    return bounded(base + 0.18 * delta_term + heuristic_bonus, 0.0, 1.0)


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return default
        return default


def _count_batch_changes(topk_scores: list[Stage2Score], candidate_actions: list[dict[str, Any]], batch_size: int) -> int:
    if not topk_scores:
        return 0
    pre_ids = {score.candidate_id for score in topk_scores[:batch_size]}
    final_scores = {
        score.candidate_id: float(score.acquisition_score)
        for score in topk_scores
    }
    for action in candidate_actions:
        final_scores[str(action["candidate_id"])] = round(final_scores[str(action["candidate_id"])] + float(action.get("score_delta", 0.0)), 6)
    ranked_ids = [
        candidate_id
        for candidate_id, _ in sorted(
            final_scores.items(),
            key=lambda item: (item[1], -next(index for index, score in enumerate(topk_scores) if score.candidate_id == item[0])),
            reverse=True,
        )
    ]
    post_ids = set(ranked_ids[:batch_size])
    return len(pre_ids.symmetric_difference(post_ids)) // 2


def _sanitize_actions(
    candidate_actions: list[dict[str, Any]],
    task: TaskRepresentation,
    topk_candidates: list[Candidate],
    topk_scores: list[Stage2Score],
    batch_size: int,
    constants: SharedRunConstants,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limits = _profile_limits(task, constants)
    score_context = _build_score_context(topk_scores, batch_size, int(limits["boundary_band"]))
    keep_map = _seed_keep_actions(topk_candidates, score_context)
    raw_map: dict[str, dict[str, Any]] = {}
    filtered_counts: Counter[str] = Counter()

    for item in candidate_actions:
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in keep_map or candidate_id not in score_context:
            filtered_counts["invalid_candidate"] += 1
            continue
        decision = str(item.get("decision", "keep")).strip().lower()
        if decision not in {"keep", "downweight", "reject", "prefer_for_exploration"}:
            decision = "keep"
        context = score_context[candidate_id]
        if decision == "keep":
            delta = 0.0
        elif decision == "downweight":
            delta = bounded(_coerce_float(item.get("score_delta", constants.s2_downweight_delta), constants.s2_downweight_delta), constants.s2_downweight_delta, -constants.s2_min_abs_delta)
        elif decision == "prefer_for_exploration":
            delta = bounded(_coerce_float(item.get("score_delta", constants.s2_prefer_delta), constants.s2_prefer_delta), constants.s2_min_abs_delta, constants.s2_prefer_delta)
        else:
            delta = bounded(_coerce_float(item.get("score_delta", constants.s2_reject_delta), constants.s2_reject_delta), constants.s2_reject_delta, -constants.s2_min_abs_delta)

        confidence = _estimate_confidence(item, decision, constants)
        if decision != "keep" and confidence < float(limits["min_nonkeep_confidence"]):
            filtered_counts["low_confidence"] += 1
            continue
        if decision == "reject" and confidence < float(limits["min_reject_confidence"]):
            filtered_counts["reject_below_threshold"] += 1
            continue
        if decision == "reject" and int(limits["max_reject_actions"]) == 0:
            filtered_counts["reject_disabled_for_profile"] += 1
            continue
        if decision == "prefer_for_exploration" and int(limits["max_prefer_actions"]) == 0:
            filtered_counts["prefer_disabled_for_profile"] += 1
            continue
        if decision in {"downweight", "reject"} and not (bool(context["inside_batch"]) or bool(context["near_boundary"])):
            filtered_counts["outside_boundary_band"] += 1
            continue
        if decision == "prefer_for_exploration":
            if bool(context["inside_batch"]) or not bool(context["near_boundary"]):
                filtered_counts["prefer_not_near_boundary"] += 1
                continue
            if float(context["predicted_std"]) < constants.s2_min_predicted_std_for_exploration:
                filtered_counts["prefer_low_uncertainty"] += 1
                continue
            if float(context["predicted_mean_gap_to_cutoff"]) < -0.015:
                filtered_counts["prefer_low_mean"] += 1
                continue
        if abs(delta) < constants.s2_min_abs_delta:
            filtered_counts["below_delta_floor"] += 1
            continue

        normalized = {
            "candidate_id": candidate_id,
            "decision": decision,
            "score_delta": round(delta, 4),
            "reason": str(item.get("reason", "")).strip() or "llm_gate",
            "confidence": round(confidence, 4),
            "rank": int(context["rank"]),
        }
        current = raw_map.get(candidate_id)
        if current is None or (
            float(normalized["confidence"]) > float(current["confidence"])
            or (
                float(normalized["confidence"]) == float(current["confidence"])
                and (
                    abs(float(normalized["score_delta"])) > abs(float(current["score_delta"]))
                    or (
                        abs(float(normalized["score_delta"])) == abs(float(current["score_delta"]))
                        and _decision_priority(str(normalized["decision"])) > _decision_priority(str(current["decision"]))
                    )
                )
            )
        ):
            raw_map[candidate_id] = normalized

    selected_nonkeep: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    for item in sorted(
        (value for value in raw_map.values() if str(value["decision"]) != "keep"),
        key=lambda value: (
            float(value["confidence"]),
            abs(float(value["score_delta"])),
            -int(value["rank"]),
            _decision_priority(str(value["decision"])),
        ),
        reverse=True,
    ):
        decision = str(item["decision"])
        if len(selected_nonkeep) >= int(limits["max_nonkeep_actions"]):
            filtered_counts["budget_nonkeep"] += 1
            continue
        if decision == "downweight" and decision_counts["downweight"] >= int(limits["max_downweight_actions"]):
            filtered_counts["budget_downweight"] += 1
            continue
        if decision == "prefer_for_exploration" and decision_counts["prefer_for_exploration"] >= int(limits["max_prefer_actions"]):
            filtered_counts["budget_prefer"] += 1
            continue
        if decision == "reject" and decision_counts["reject"] >= int(limits["max_reject_actions"]):
            filtered_counts["budget_reject"] += 1
            continue
        selected_nonkeep.append(item)
        decision_counts[decision] += 1

    reverted_for_batch_change_limit = 0
    while selected_nonkeep and _count_batch_changes(topk_scores, selected_nonkeep, batch_size) > int(limits["max_final_batch_changes"]):
        weakest_index = min(
            range(len(selected_nonkeep)),
            key=lambda index: (
                float(selected_nonkeep[index]["confidence"]),
                abs(float(selected_nonkeep[index]["score_delta"])),
                _decision_priority(str(selected_nonkeep[index]["decision"])),
                -int(selected_nonkeep[index]["rank"]),
            ),
        )
        selected_nonkeep.pop(weakest_index)
        reverted_for_batch_change_limit += 1

    final_map = keep_map
    for item in selected_nonkeep:
        final_map[str(item["candidate_id"])] = item

    ordered_actions = [final_map[candidate.candidate_id] for candidate in topk_candidates]
    final_counts = Counter(str(item["decision"]) for item in ordered_actions)
    selected_confidences = [float(item["confidence"]) for item in selected_nonkeep]
    diagnostics = {
        "topk_size": len(topk_candidates),
        "action_counts": dict(final_counts),
        "nonkeep_action_count": len(selected_nonkeep),
        "raw_nonkeep_suggestion_count": len([value for value in raw_map.values() if str(value["decision"]) != "keep"]),
        "mean_nonkeep_confidence": round(mean(selected_confidences), 4) if selected_confidences else 0.0,
        "min_nonkeep_confidence_threshold": float(limits["min_nonkeep_confidence"]),
        "min_reject_confidence_threshold": float(limits["min_reject_confidence"]),
        "max_final_batch_changes": int(limits["max_final_batch_changes"]),
        "estimated_final_batch_changes": _count_batch_changes(topk_scores, selected_nonkeep, batch_size),
        "filtered_counts": dict(filtered_counts),
        "reverted_for_batch_change_limit": reverted_for_batch_change_limit,
        "profile_caution_mode": "hindered_stage2_mean_protection" if bool(limits["is_hindered"]) else "standard_sparse_gate",
        "selected_nonkeep_confidences": [round(value, 4) for value in selected_confidences],
    }
    return ordered_actions, diagnostics


def _fallback_gate(
    task: TaskRepresentation,
    summary: LLMS2Summary,
    aggregate: RoundAggregate,
    topk_candidates: list[Candidate],
    topk_scores: list[Stage2Score],
    batch_size: int,
    constants: SharedRunConstants,
) -> LLMS2Gate:
    poor_values: dict[str, set[str]] = {}
    strong_values: dict[str, str] = {}
    for axis, table in aggregate.factor_tables.items():
        if table:
            strong_values[axis] = str(table[0]["label"])
            poor_values[axis] = {
                str(row["label"])
                for row in table
                if row["count"] >= 3 and float(row["mean_yield"]) < aggregate.mean_yield - 0.06
            }

    cutoff_index = min(max(batch_size - 1, 0), len(topk_scores) - 1) if topk_scores else 0
    cutoff_mean = float(topk_scores[cutoff_index].predicted_mean) if topk_scores else 0.0
    raw_actions: list[dict[str, Any]] = []
    for rank, (candidate, score) in enumerate(zip(topk_candidates, topk_scores), start=1):
        candidate_labels = {axis: str(candidate.parameters.get(axis, "")) for axis in ("ligand_label", "base_label", "solvent_label")}
        poor_hits = [axis for axis in poor_values if candidate_labels.get(axis, "") in poor_values.get(axis, set())]
        if poor_hits and rank <= batch_size + 1 and float(score.predicted_mean) <= cutoff_mean + 0.01:
            confidence = 0.9 if len(poor_hits) >= 2 else 0.82
            raw_actions.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "decision": "downweight",
                    "score_delta": constants.s2_downweight_delta,
                    "reason": f"consistent underperforming observed factor: {','.join(sorted(poor_hits))}",
                    "confidence": confidence,
                }
            )
            continue
        if (
            task.profile_name != "hindered_aryl_chloride_activation"
            and rank > batch_size
            and rank <= batch_size + 2
            and float(score.predicted_std) >= constants.s2_min_predicted_std_for_exploration + 0.02
            and float(score.predicted_mean) >= cutoff_mean - 0.01
            and candidate_labels.get("solvent_label", "") != strong_values.get("solvent_label", "")
        ):
            raw_actions.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "decision": "prefer_for_exploration",
                    "score_delta": constants.s2_prefer_delta,
                    "reason": "narrow exploratory swap near BO boundary",
                    "confidence": 0.8,
                }
            )
    ordered_actions, diagnostics = _sanitize_actions(raw_actions, task, topk_candidates, topk_scores, batch_size, constants)
    return LLMS2Gate(
        invoked=False,
        status="fallback",
        model="deterministic_backstop",
        summary=summary.summary,
        candidate_actions=ordered_actions,
        diagnostics=diagnostics,
        raw_response={},
    )


def rerank_topk(
    task: TaskRepresentation,
    aggregate: RoundAggregate,
    summary: LLMS2Summary,
    topk_candidates: list[Candidate],
    topk_scores: list[Stage2Score],
    batch_size: int,
    client,
    constants: SharedRunConstants,
) -> LLMS2Gate:
    fallback = _fallback_gate(task, summary, aggregate, topk_candidates, topk_scores, batch_size, constants)
    payload = {
        "task_profile": task.profile_name,
        "round_summary": {
            "summary": summary.summary,
            "promising_patterns": summary.promising_patterns,
            "risky_patterns": summary.risky_patterns,
            "exploration_axes": summary.exploration_axes,
            "mean_yield": aggregate.mean_yield,
            "best_yield": aggregate.best_yield,
        },
        "guidance": {
            "default_action": "keep",
            "sparse_correction_layer": True,
            "prefer_few_high_confidence_changes": True,
            "max_nonkeep_actions": constants.s2_max_nonkeep_actions,
            "max_final_batch_changes": constants.s2_max_final_batch_changes,
            "protect_stage2_batch_mean": True,
            "hindered_profile_requires_extra_caution": task.profile_name == "hindered_aryl_chloride_activation",
        },
        "topk_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "parameters": candidate.parameters,
                "predicted_mean": score.predicted_mean,
                "predicted_std": score.predicted_std,
                "acquisition_score": score.acquisition_score,
                "rank": rank,
            }
            for rank, (candidate, score) in enumerate(zip(topk_candidates, topk_scores), start=1)
        ],
        "output_schema": {
            "summary": "short string",
            "candidate_actions": [
                {
                    "candidate_id": "id",
                    "decision": "keep|downweight|reject|prefer_for_exploration",
                    "score_delta": "-0.08 to 0.02",
                    "confidence": "0.0-1.0",
                    "reason": "short reason",
                }
            ],
        },
    }
    response = client.complete_json(
        "topk_rerank_gate",
        (
            "You are the LLM-S2 BO rerank gate. Return JSON only. "
            "Default to keep. Suggest only a very sparse set of high-confidence corrections. "
            "Protect stage2 batch mean, especially for hindered profiles. "
            "Do not behave like a second optimizer and do not output the final batch directly."
        ),
        payload,
    )
    if response["status"] != "success":
        return LLMS2Gate(
            invoked=bool(response["invoked"]),
            status=str(response["status"]),
            model=str(response["model"]),
            summary=fallback.summary,
            candidate_actions=fallback.candidate_actions,
            diagnostics=fallback.diagnostics,
            raw_response=response,
        )
    ordered_actions, diagnostics = _sanitize_actions(
        [item for item in response["parsed"].get("candidate_actions", []) if isinstance(item, dict)],
        task,
        topk_candidates,
        topk_scores,
        batch_size,
        constants,
    )
    return LLMS2Gate(
        invoked=True,
        status="success",
        model=str(response["model"]),
        summary=str(response["parsed"].get("summary", "")).strip() or summary.summary,
        candidate_actions=ordered_actions,
        diagnostics=diagnostics,
        raw_response=response,
    )

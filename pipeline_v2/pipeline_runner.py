from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

from pipeline_v2.config import DATASET_ROOT, FlowConfig
from pipeline_v2.llm_modules.client import OpenRouterJSONClient
from pipeline_v2.llm_modules.precedent_critic import calibrate_precedent_critique, run_precedent_critic
from pipeline_v2.llm_modules.result_summarizer import summarize_results
from pipeline_v2.llm_modules.topk_rerank_gate import rerank_topk
from pipeline_v2.reporting import build_calibration_benchmark, build_calibration_round, build_final_benchmark, build_smoke_comparison
from pipeline_v2.shared.candidate_schema import Candidate, LLMS1Critique, LLMS2Gate, LLMS2Summary, Observation, Stage2Score
from pipeline_v2.shared.io_utils import ensure_dir, mean, utc_now_iso, write_csv, write_json, write_markdown
from pipeline_v2.shared.offline_simulator import compute_oracle_summary, simulate_observation
from pipeline_v2.shared.task_representation import build_task_representation
from pipeline_v2.stage1.candidate_space_builder import build_candidate_universe
from pipeline_v2.stage1.initial_batch_designer import design_initial_batch
from pipeline_v2.stage1.retrieval_initializer import load_historical_dataset, retrieve_support_bundle
from pipeline_v2.stage1.rule_filter import apply_rule_filter
from pipeline_v2.stage1.transfer_scorer import score_candidates, summarize_transfer_axes
from pipeline_v2.stage2.optimizer_loop import MixedBatchBO
from pipeline_v2.stage2.result_aggregator import aggregate_round
from pipeline_v2.stage2.trust_region import build_trust_region

def _build_client(flow_config: FlowConfig) -> OpenRouterJSONClient:
    return OpenRouterJSONClient(
        base_url=flow_config.llm.base_url,
        timeout_seconds=flow_config.llm.timeout_seconds,
        model_fallbacks=flow_config.llm.model_fallbacks,
    )


def _candidate_rows(flow_name: str, protocol_name: str, profile_name: str, candidates: list[Candidate]) -> list[dict[str, object]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "flow_name": flow_name,
                "protocol": protocol_name,
                "profile_name": profile_name,
                "candidate_id": candidate.candidate_id,
                "stage1_rank": candidate.stage1_rank,
                "stage1_score": candidate.stage1_score,
                "feasible": candidate.feasible,
                "rejection_reasons": "; ".join(candidate.rejection_reasons),
                "llm_stage1_delta": candidate.llm_stage1_delta,
                "llm_stage1_flags": "; ".join(candidate.llm_stage1_flags),
                "ligand_label": candidate.parameters.get("ligand_label", ""),
                "base_label": candidate.parameters.get("base_label", ""),
                "solvent_label": candidate.parameters.get("solvent_label", ""),
                "temperature_c": candidate.parameters.get("temperature_c", ""),
                "catalyst_mol_pct": candidate.parameters.get("catalyst_mol_pct", ""),
                "base_equiv": candidate.parameters.get("base_equiv", ""),
                **candidate.stage1_components,
            }
        )
    return rows


def _observation_rows(flow_name: str, protocol_name: str, profile_name: str, observations: list[Observation]) -> list[dict[str, object]]:
    rows = []
    for observation in observations:
        rows.append(
            {
                "flow_name": flow_name,
                "protocol": protocol_name,
                "profile_name": profile_name,
                "round_index": observation.round_index,
                "candidate_id": observation.candidate_id,
                "yield_value": observation.yield_value,
                "selectivity": observation.selectivity,
                "outcome_label": observation.outcome_label,
                "ligand_label": observation.parameters.get("ligand_label", ""),
                "base_label": observation.parameters.get("base_label", ""),
                "solvent_label": observation.parameters.get("solvent_label", ""),
                "temperature_c": observation.parameters.get("temperature_c", ""),
                "catalyst_mol_pct": observation.parameters.get("catalyst_mol_pct", ""),
                "base_equiv": observation.parameters.get("base_equiv", ""),
            }
        )
    return rows


def _stage2_rows(flow_name: str, protocol_name: str, profile_name: str, round_index: int, scores: list[Stage2Score], candidate_by_id: dict[str, Candidate]) -> list[dict[str, object]]:
    rows = []
    for score in scores:
        candidate = candidate_by_id[score.candidate_id]
        rows.append(
            {
                "flow_name": flow_name,
                "protocol": protocol_name,
                "profile_name": profile_name,
                "round_index": round_index,
                "candidate_id": score.candidate_id,
                "predicted_mean": score.predicted_mean,
                "predicted_std": score.predicted_std,
                "acquisition_score": score.acquisition_score,
                "prior_score": score.prior_score,
                "trust_region_bonus": score.trust_region_bonus,
                "llm_delta": score.llm_delta,
                "llm_decision": score.llm_decision,
                "llm_reason": score.llm_reason,
                "final_score": score.final_score,
                "ligand_label": candidate.parameters.get("ligand_label", ""),
                "base_label": candidate.parameters.get("base_label", ""),
                "solvent_label": candidate.parameters.get("solvent_label", ""),
                "temperature_c": candidate.parameters.get("temperature_c", ""),
            }
        )
    return rows


def _apply_gate(scores: list[Stage2Score], gate: LLMS2Gate) -> list[Stage2Score]:
    actions = {str(item["candidate_id"]): item for item in gate.candidate_actions}
    adjusted: list[Stage2Score] = []
    for score in scores:
        action = actions.get(score.candidate_id)
        if action:
            delta = float(action.get("score_delta", 0.0))
            decision = str(action.get("decision", "keep"))
            score.llm_delta = round(delta, 4)
            score.llm_decision = decision
            score.llm_reason = str(action.get("reason", ""))
            score.final_score = round(score.acquisition_score + delta, 4)
        adjusted.append(score)
    adjusted.sort(key=lambda item: item.final_score, reverse=True)
    return adjusted


def _extract_stage1_reason_labels(candidate: Candidate) -> set[str]:
    labels: set[str] = set()
    for flag in candidate.llm_stage1_flags:
        parts = str(flag).split(":", 2)
        if len(parts) >= 3:
            labels.add(parts[1])
    for reason in candidate.rejection_reasons:
        parts = str(reason).split(":", 2)
        if len(parts) >= 3 and parts[0] == "llm_reject_value":
            labels.add(parts[1])
        elif len(parts) >= 2 and parts[0] == "llm_reject_combination":
            labels.add(parts[1])
    return labels


def _build_stage1_diagnostics(
    llm_critique: LLMS1Critique | None,
    base_universe_size: int,
    scored_candidates: list[Candidate],
    feasible_candidates: list[Candidate],
    rejected_candidates: list[Candidate],
    initial_batch_ids: set[str],
    baseline_stage1_ids: set[str],
    baseline_feasible_count: int,
) -> dict[str, object]:
    if llm_critique is None:
        return {
            "enabled": False,
            "candidate_pool_size_before_s1": base_universe_size,
            "candidate_pool_size_after_s1": len(scored_candidates),
            "candidate_pool_size_after_rule_filter": len(feasible_candidates),
            "baseline_candidate_pool_after_rule_filter": baseline_feasible_count,
            "initial_batch_size": len(initial_batch_ids),
            "final_action_counts": {"rejected": 0, "downweighted": 0, "bonused_preferred": 0, "unchanged": len(scored_candidates)},
            "reason_label_counts": {},
            "contradiction_resolution_count": 0,
            "initial_batch_symmetric_difference_vs_backbone": 0,
            "initial_batch_changed_items_vs_backbone": 0,
        }

    llm_rejected_ids = {
        candidate.candidate_id
        for candidate in rejected_candidates
        if any(reason.startswith("llm_reject_") for reason in candidate.rejection_reasons)
    }
    final_action_counts: Counter[str] = Counter()
    reason_label_counts: Counter[str] = Counter()
    for candidate in scored_candidates:
        if candidate.candidate_id in llm_rejected_ids:
            final_action_counts["rejected"] += 1
        elif candidate.llm_stage1_delta < 0:
            final_action_counts["downweighted"] += 1
        elif candidate.llm_stage1_delta > 0:
            final_action_counts["bonused_preferred"] += 1
        else:
            final_action_counts["unchanged"] += 1
        for label in _extract_stage1_reason_labels(candidate):
            reason_label_counts[label] += 1

    batch_symmetric_diff = len(initial_batch_ids.symmetric_difference(baseline_stage1_ids)) if baseline_stage1_ids else 0
    return {
        "enabled": True,
        "candidate_pool_size_before_s1": base_universe_size,
        "candidate_pool_size_after_s1": len(scored_candidates),
        "candidate_pool_size_after_rule_filter": len(feasible_candidates),
        "baseline_candidate_pool_after_rule_filter": baseline_feasible_count,
        "initial_batch_size": len(initial_batch_ids),
        "final_action_counts": dict(final_action_counts),
        "reason_label_counts": dict(reason_label_counts),
        "contradiction_resolution_count": int(llm_critique.diagnostics.get("contradiction_resolution_count", 0)),
        "estimated_modified_candidate_count": int(llm_critique.diagnostics.get("estimated_modified_candidate_count", 0)),
        "estimated_modified_fraction": float(llm_critique.diagnostics.get("estimated_modified_fraction", 0.0)),
        "estimated_reject_candidate_count": int(llm_critique.diagnostics.get("estimated_reject_candidate_count", 0)),
        "dropped_for_coverage_count": int(llm_critique.diagnostics.get("dropped_for_coverage_count", 0)),
        "dropped_for_budget_count": int(llm_critique.diagnostics.get("dropped_for_budget_count", 0)),
        "reject_downgrade_count": int(llm_critique.diagnostics.get("reject_downgrade_count", 0)),
        "initial_batch_symmetric_difference_vs_backbone": batch_symmetric_diff,
        "initial_batch_changed_items_vs_backbone": batch_symmetric_diff // 2,
        "raw_diagnostics": dict(llm_critique.diagnostics),
    }


def _build_stage2_diagnostics(gate: LLMS2Gate | None, topk_size: int, batch_symmetric_diff: int) -> dict[str, object]:
    if gate is None:
        return {
            "enabled": False,
            "topk_size": topk_size,
            "action_counts": {"keep": topk_size},
            "nonkeep_action_count": 0,
            "mean_nonkeep_confidence": 0.0,
            "max_final_batch_changes": 0,
            "estimated_final_batch_changes": 0,
            "final_bo_batch_symmetric_difference": 0,
            "final_bo_batch_changed_items": 0,
        }
    return {
        "enabled": True,
        "topk_size": topk_size,
        "action_counts": dict(gate.diagnostics.get("action_counts", {})),
        "nonkeep_action_count": int(gate.diagnostics.get("nonkeep_action_count", 0)),
        "raw_nonkeep_suggestion_count": int(gate.diagnostics.get("raw_nonkeep_suggestion_count", 0)),
        "mean_nonkeep_confidence": float(gate.diagnostics.get("mean_nonkeep_confidence", 0.0)),
        "min_nonkeep_confidence_threshold": float(gate.diagnostics.get("min_nonkeep_confidence_threshold", 0.0)),
        "max_final_batch_changes": int(gate.diagnostics.get("max_final_batch_changes", 0)),
        "estimated_final_batch_changes": int(gate.diagnostics.get("estimated_final_batch_changes", 0)),
        "final_bo_batch_symmetric_difference": batch_symmetric_diff,
        "final_bo_batch_changed_items": batch_symmetric_diff // 2,
        "raw_diagnostics": dict(gate.diagnostics),
    }


def _action_count(mapping: dict[str, object], key: str) -> int:
    return int(mapping.get(key, 0))


def _reason_count(mapping: dict[str, object], key: str) -> int:
    return int(mapping.get(key, 0))


def _candidate_mean_yield(candidates: list[Candidate], task, seed: int, round_index: int) -> float:
    if not candidates:
        return 0.0
    return round(mean([simulate_observation(candidate, task, seed, round_index).yield_value for candidate in candidates]), 4)


def _report_lines(flow_config: FlowConfig, protocol_name: str, summary_rows: list[dict[str, object]]) -> list[str]:
    lines = [f"# {flow_config.flow_name}: {protocol_name}", "",
             "Reaction-condition optimization evaluated with the offline simulator.", "",
             "| Profile | Best observed | Oracle | Stage 1 changes | Stage 2 changes |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary_rows:
        lines.append(f"| {row['profile_name']} | {row['best_observed_yield']} | {row['oracle_best_yield']} | "
                     f"{row['llm_s1_batch_delta_count']} | {row['llm_s2_batch_delta_count']} |")
    return lines


def _run_profile(flow_config: FlowConfig, protocol, profile_name: str, output_dir: Path, dataset, client) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    constants = flow_config.constants
    task = build_task_representation(profile_name, constants.reaction_family, constants.objective)
    profile_dir = ensure_dir(output_dir / profile_name)
    write_json(profile_dir / "task.json", task)

    support = retrieve_support_bundle(task, dataset, constants.precedent_pool_size, constants.support_case_pool_size)
    write_csv(profile_dir / "retrieval_table.csv", list(support.retrieval_table))

    base_universe = build_candidate_universe(task, support, dataset, constants.candidate_pool_cap)
    oracle_summary = compute_oracle_summary(base_universe, task, protocol.seed)
    write_json(profile_dir / "oracle_summary.json", oracle_summary)

    llm_critique: LLMS1Critique | None = None
    llm_s1_batch_delta_count = 0
    llm_s1_rejected_count = 0
    baseline_stage1_ids: set[str] = set()
    baseline_feasible_count = 0
    if flow_config.llm_stage1_enabled:
        baseline_candidates = copy.deepcopy(base_universe)
        baseline_scored = score_candidates(task, baseline_candidates, support, None)
        baseline_feasible, _ = apply_rule_filter(task, baseline_scored, None)
        baseline_feasible_count = len(baseline_feasible)
        baseline_batch, _ = design_initial_batch(baseline_feasible, protocol.initial_batch_size)
        baseline_stage1_ids = {candidate.candidate_id for candidate in baseline_batch}
        raw_llm_critique = run_precedent_critic(task, support, client, constants)
        write_json(profile_dir / "llm_s1_raw_critique.json", raw_llm_critique)
        llm_critique = calibrate_precedent_critique(raw_llm_critique, base_universe, constants)
        write_json(profile_dir / "llm_s1_critique.json", llm_critique)

    candidates = copy.deepcopy(base_universe)
    scored_candidates = score_candidates(task, candidates, support, llm_critique)
    feasible_candidates, rejected_candidates = apply_rule_filter(task, scored_candidates, llm_critique)
    initial_batch, stage1_trace = design_initial_batch(feasible_candidates, protocol.initial_batch_size)
    initial_batch_ids = {candidate.candidate_id for candidate in initial_batch}
    if not flow_config.llm_stage1_enabled:
        baseline_feasible_count = len(feasible_candidates)
        baseline_stage1_ids = set(initial_batch_ids)

    stage1_diagnostics = _build_stage1_diagnostics(
        llm_critique=llm_critique,
        base_universe_size=len(base_universe),
        scored_candidates=scored_candidates,
        feasible_candidates=feasible_candidates,
        rejected_candidates=rejected_candidates,
        initial_batch_ids=initial_batch_ids,
        baseline_stage1_ids=baseline_stage1_ids,
        baseline_feasible_count=baseline_feasible_count,
    )
    write_json(profile_dir / "stage1_diagnostics.json", stage1_diagnostics)
    if flow_config.llm_stage1_enabled:
        write_json(profile_dir / "llm_s1_diagnostics.json", stage1_diagnostics)
        llm_s1_batch_delta_count = int(stage1_diagnostics["initial_batch_symmetric_difference_vs_backbone"])
        llm_s1_rejected_count = _action_count(stage1_diagnostics["final_action_counts"], "rejected")

    candidate_by_id = {candidate.candidate_id: candidate for candidate in feasible_candidates}
    write_csv(profile_dir / "candidate_universe.csv", _candidate_rows(flow_config.flow_name, protocol.name, profile_name, scored_candidates))
    write_csv(profile_dir / "rejected_candidates.csv", _candidate_rows(flow_config.flow_name, protocol.name, profile_name, rejected_candidates))
    write_csv(profile_dir / "stage1_selection_trace.csv", list(stage1_trace))
    write_json(profile_dir / "transfer_axis_summary.json", summarize_transfer_axes(feasible_candidates))
    write_csv(profile_dir / "initial_batch.csv", _candidate_rows(flow_config.flow_name, protocol.name, profile_name, initial_batch))

    round_rows: list[dict[str, object]] = []
    experiment_rows: list[dict[str, object]] = []
    all_observations: list[Observation] = [simulate_observation(candidate, task, protocol.seed, 1) for candidate in initial_batch]
    experiment_rows.extend(_observation_rows(flow_config.flow_name, protocol.name, profile_name, all_observations))
    write_csv(profile_dir / "round1_observations.csv", _observation_rows(flow_config.flow_name, protocol.name, profile_name, all_observations))
    stage1_aggregate = aggregate_round(all_observations, 1)
    write_json(profile_dir / "round1_aggregate.json", stage1_aggregate)
    stage2_batch_mean_yields: list[float] = []
    stage2_batch_best_yields: list[float] = []
    stage2_topk_true_means: list[float] = []
    stage2_pre_gate_true_means: list[float] = []
    stage2_post_gate_true_means: list[float] = []
    stage2_gate_quality_deltas: list[float] = []
    stage2_mean_nonkeep_confidences: list[float] = []
    round_rows.append(
        {
            "flow_name": flow_config.flow_name,
            "protocol": protocol.name,
            "profile_name": profile_name,
            "round_index": 1,
            "batch_kind": "stage1_initial_96",
            "batch_size": len(all_observations),
            "batch_mean_yield": stage1_aggregate.mean_yield,
            "batch_best_yield": stage1_aggregate.best_yield,
            "mean_selectivity": stage1_aggregate.mean_selectivity,
            "cumulative_best_yield": stage1_aggregate.best_yield,
        }
    )

    llm_s2_batch_delta_count = 0
    llm_s2_action_count = 0
    llm_s2_status = "not_enabled"
    llm_s2_keep_count = 0
    llm_s2_downweight_count = 0
    llm_s2_reject_count = 0
    llm_s2_prefer_count = 0
    llm_s2_changed_items = 0
    optimizer = MixedBatchBO(constants.acquisition_beta, constants.diversity_weight, constants.prior_blend)
    for offset in range(protocol.stage2_rounds):
        round_index = 2 + offset
        trust_region = build_trust_region(all_observations)
        write_json(profile_dir / f"round{round_index}_trust_region.json", trust_region)
        observed_ids = {observation.candidate_id for observation in all_observations}
        ranked_scores = optimizer.rank_candidates(feasible_candidates, all_observations, trust_region, observed_ids)
        topk_scores = copy.deepcopy(ranked_scores[: constants.topk_rerank_pool_size])
        topk_candidates = [candidate_by_id[score.candidate_id] for score in topk_scores]
        write_csv(profile_dir / f"round{round_index}_topk_before_gate.csv", _stage2_rows(flow_config.flow_name, protocol.name, profile_name, round_index, topk_scores, candidate_by_id))

        pre_gate_selected, pre_gate_trace = optimizer.select_diverse_batch(copy.deepcopy(topk_scores), candidate_by_id, protocol.stage2_batch_size)
        adjusted_scores = copy.deepcopy(topk_scores)
        llm_gate: LLMS2Gate | None = None
        if flow_config.llm_stage2_enabled:
            pre_selection_aggregate = aggregate_round(all_observations, round_index - 1)
            llm_summary: LLMS2Summary = summarize_results(task, pre_selection_aggregate, client)
            llm_gate = rerank_topk(task, pre_selection_aggregate, llm_summary, topk_candidates, adjusted_scores, protocol.stage2_batch_size, client, constants)
            adjusted_scores = _apply_gate(adjusted_scores, llm_gate)
            llm_s2_status = llm_gate.status
            write_json(profile_dir / f"round{round_index}_llm_s2_summary.json", llm_summary)
            write_json(profile_dir / f"round{round_index}_llm_s2_gate.json", llm_gate)
        post_gate_selected, post_gate_trace = optimizer.select_diverse_batch(adjusted_scores, candidate_by_id, protocol.stage2_batch_size)
        write_csv(profile_dir / f"round{round_index}_topk_after_gate.csv", _stage2_rows(flow_config.flow_name, protocol.name, profile_name, round_index, adjusted_scores, candidate_by_id))
        write_csv(profile_dir / f"round{round_index}_selection_trace_before_gate.csv", list(pre_gate_trace))
        write_csv(profile_dir / f"round{round_index}_selection_trace_after_gate.csv", list(post_gate_trace))
        batch_symmetric_diff = len({candidate.candidate_id for candidate in pre_gate_selected}.symmetric_difference({candidate.candidate_id for candidate in post_gate_selected}))
        stage2_diagnostics = _build_stage2_diagnostics(llm_gate if flow_config.llm_stage2_enabled else None, len(topk_candidates), batch_symmetric_diff)
        topk_true_mean = _candidate_mean_yield(topk_candidates, task, protocol.seed, round_index)
        pre_gate_true_mean = _candidate_mean_yield(pre_gate_selected, task, protocol.seed, round_index)
        post_gate_true_mean = _candidate_mean_yield(post_gate_selected, task, protocol.seed, round_index)
        stage2_diagnostics["topk_true_mean_yield"] = topk_true_mean
        stage2_diagnostics["pre_gate_batch_true_mean_yield"] = pre_gate_true_mean
        stage2_diagnostics["post_gate_batch_true_mean_yield"] = post_gate_true_mean
        stage2_diagnostics["gate_batch_quality_delta"] = round(post_gate_true_mean - pre_gate_true_mean, 4)
        write_json(profile_dir / f"round{round_index}_stage2_diagnostics.json", stage2_diagnostics)
        if flow_config.llm_stage2_enabled:
            write_json(profile_dir / f"round{round_index}_llm_s2_diagnostics.json", stage2_diagnostics)
            llm_s2_batch_delta_count += batch_symmetric_diff
            llm_s2_action_count += int(stage2_diagnostics.get("nonkeep_action_count", 0))
            action_counts = dict(stage2_diagnostics.get("action_counts", {}))
            llm_s2_keep_count += _action_count(action_counts, "keep")
            llm_s2_downweight_count += _action_count(action_counts, "downweight")
            llm_s2_reject_count += _action_count(action_counts, "reject")
            llm_s2_prefer_count += _action_count(action_counts, "prefer_for_exploration")
            llm_s2_changed_items += int(stage2_diagnostics.get("final_bo_batch_changed_items", 0))
            stage2_mean_nonkeep_confidences.append(float(stage2_diagnostics.get("mean_nonkeep_confidence", 0.0)))
        new_observations = [simulate_observation(candidate, task, protocol.seed, round_index) for candidate in post_gate_selected]
        all_observations.extend(new_observations)
        experiment_rows.extend(_observation_rows(flow_config.flow_name, protocol.name, profile_name, new_observations))
        write_csv(profile_dir / f"round{round_index}_observations.csv", _observation_rows(flow_config.flow_name, protocol.name, profile_name, new_observations))
        batch_aggregate = aggregate_round(new_observations, round_index)
        write_json(profile_dir / f"round{round_index}_aggregate.json", batch_aggregate)
        stage2_batch_mean_yields.append(batch_aggregate.mean_yield)
        stage2_batch_best_yields.append(batch_aggregate.best_yield)
        stage2_topk_true_means.append(topk_true_mean)
        stage2_pre_gate_true_means.append(pre_gate_true_mean)
        stage2_post_gate_true_means.append(post_gate_true_mean)
        stage2_gate_quality_deltas.append(round(post_gate_true_mean - pre_gate_true_mean, 4))
        round_rows.append(
            {
                "flow_name": flow_config.flow_name,
                "protocol": protocol.name,
                "profile_name": profile_name,
                "round_index": round_index,
                "batch_kind": "stage2_bo_batch",
                "batch_size": len(new_observations),
                "batch_mean_yield": batch_aggregate.mean_yield,
                "batch_best_yield": batch_aggregate.best_yield,
                "mean_selectivity": batch_aggregate.mean_selectivity,
                "cumulative_best_yield": round(max(observation.yield_value for observation in all_observations), 4),
                "topk_true_mean_yield": topk_true_mean,
                "pre_gate_batch_true_mean_yield": pre_gate_true_mean,
                "post_gate_batch_true_mean_yield": post_gate_true_mean,
                "gate_batch_quality_delta": round(post_gate_true_mean - pre_gate_true_mean, 4),
            }
        )

    best_observed_yield = round(max(observation.yield_value for observation in all_observations), 4) if all_observations else 0.0
    oracle_best_yield = float(oracle_summary["oracle_best_yield"])
    stage1_action_counts = dict(stage1_diagnostics.get("final_action_counts", {}))
    stage1_reason_counts = dict(stage1_diagnostics.get("reason_label_counts", {}))
    summary_row = {
        "flow_name": flow_config.flow_name,
        "protocol": protocol.name,
        "profile_name": profile_name,
        "run_success": bool(initial_batch) and (not flow_config.llm_stage2_enabled or llm_s2_status != "not_enabled"),
        "total_experiments": len(all_observations),
        "best_observed_yield": best_observed_yield,
        "mean_observed_yield": round(mean([observation.yield_value for observation in all_observations]), 4) if all_observations else 0.0,
        "oracle_best_yield": oracle_best_yield,
        "regret_to_oracle": round(oracle_best_yield - best_observed_yield, 4),
        "cumulative_regret": round(sum(oracle_best_yield - observation.yield_value for observation in all_observations), 4),
        "llm_s1_status": llm_critique.status if llm_critique else "disabled",
        "llm_s1_nonzero_candidate_count": sum(1 for candidate in scored_candidates if abs(candidate.llm_stage1_delta) > 1e-9),
        "llm_s1_modified_candidate_count": int(stage1_diagnostics.get("estimated_modified_candidate_count", 0)),
        "llm_s1_modified_fraction": float(stage1_diagnostics.get("estimated_modified_fraction", 0.0)),
        "llm_s1_batch_delta_count": llm_s1_batch_delta_count,
        "llm_s1_rejected_count": llm_s1_rejected_count,
        "llm_s1_downweighted_count": _action_count(stage1_action_counts, "downweighted"),
        "llm_s1_bonused_count": _action_count(stage1_action_counts, "bonused_preferred"),
        "llm_s1_unchanged_count": _action_count(stage1_action_counts, "unchanged"),
        "llm_s1_contradiction_resolution_count": int(stage1_diagnostics.get("contradiction_resolution_count", 0)),
        "llm_s1_reason_conflict_count": _reason_count(stage1_reason_counts, "conflict"),
        "llm_s1_reason_weak_transfer_count": _reason_count(stage1_reason_counts, "weak_transfer"),
        "llm_s1_reason_dubious_compatibility_count": _reason_count(stage1_reason_counts, "dubious_compatibility"),
        "llm_s1_reason_low_support_count": _reason_count(stage1_reason_counts, "low_support"),
        "llm_s1_reason_positive_evidence_count": _reason_count(stage1_reason_counts, "positive_evidence"),
        "stage1_candidate_pool_before_s1": int(stage1_diagnostics.get("candidate_pool_size_before_s1", len(base_universe))),
        "stage1_candidate_pool_after_s1": int(stage1_diagnostics.get("candidate_pool_size_after_s1", len(scored_candidates))),
        "stage1_candidate_pool_after_rule_filter": int(stage1_diagnostics.get("candidate_pool_size_after_rule_filter", len(feasible_candidates))),
        "stage1_baseline_candidate_pool_after_rule_filter": int(stage1_diagnostics.get("baseline_candidate_pool_after_rule_filter", baseline_feasible_count)),
        "stage1_initial_batch_changed_items": int(stage1_diagnostics.get("initial_batch_changed_items_vs_backbone", 0)),
        "round1_batch_mean_yield": stage1_aggregate.mean_yield,
        "round1_batch_best_yield": stage1_aggregate.best_yield,
        "stage2_batch_mean_yield": round(mean(stage2_batch_mean_yields), 4) if stage2_batch_mean_yields else 0.0,
        "stage2_batch_best_yield": round(mean(stage2_batch_best_yields), 4) if stage2_batch_best_yields else 0.0,
        "final_stage2_batch_mean_yield": stage2_batch_mean_yields[-1] if stage2_batch_mean_yields else 0.0,
        "stage2_topk_true_mean_yield": round(mean(stage2_topk_true_means), 4) if stage2_topk_true_means else 0.0,
        "stage2_pre_gate_batch_true_mean_yield": round(mean(stage2_pre_gate_true_means), 4) if stage2_pre_gate_true_means else 0.0,
        "stage2_post_gate_batch_true_mean_yield": round(mean(stage2_post_gate_true_means), 4) if stage2_post_gate_true_means else 0.0,
        "stage2_gate_batch_quality_delta": round(mean(stage2_gate_quality_deltas), 4) if stage2_gate_quality_deltas else 0.0,
        "llm_s2_status": llm_s2_status,
        "llm_s2_action_count": llm_s2_action_count,
        "llm_s2_batch_delta_count": llm_s2_batch_delta_count,
        "llm_s2_keep_count": llm_s2_keep_count,
        "llm_s2_downweight_count": llm_s2_downweight_count,
        "llm_s2_reject_count": llm_s2_reject_count,
        "llm_s2_prefer_count": llm_s2_prefer_count,
        "llm_s2_changed_batch_items": llm_s2_changed_items,
        "llm_s2_topk_size": constants.topk_rerank_pool_size if protocol.stage2_rounds else 0,
        "llm_s2_mean_nonkeep_confidence": round(mean(stage2_mean_nonkeep_confidences), 4) if stage2_mean_nonkeep_confidences else 0.0,
        "created_at": utc_now_iso(),
    }
    return round_rows, experiment_rows, summary_row


def run_flow(flow_config: FlowConfig) -> None:
    dataset = load_historical_dataset(str(DATASET_ROOT))
    client = _build_client(flow_config)
    for protocol in flow_config.protocol_specs:
        output_dir = flow_config.output_dir_for_protocol(protocol.name)
        ensure_dir(output_dir)
        protocol_round_rows: list[dict[str, object]] = []
        protocol_experiment_rows: list[dict[str, object]] = []
        protocol_summary_rows: list[dict[str, object]] = []
        for profile_name in protocol.profiles:
            round_rows, experiment_rows, summary_row = _run_profile(flow_config, protocol, profile_name, output_dir, dataset, client)
            protocol_round_rows.extend(round_rows)
            protocol_experiment_rows.extend(experiment_rows)
            protocol_summary_rows.append(summary_row)
        write_csv(output_dir / "round_metrics.csv", protocol_round_rows)
        write_csv(output_dir / "experiments.csv", protocol_experiment_rows)
        write_csv(output_dir / "run_summary.csv", protocol_summary_rows)
        write_markdown(output_dir / "run_report.md", _report_lines(flow_config, protocol.name, protocol_summary_rows))
    build_smoke_comparison(flow_config.smoke_output_dir.parent)
    build_calibration_round(flow_config.smoke_output_dir.parent)
    build_calibration_benchmark(flow_config.smoke_output_dir.parent)
    build_final_benchmark(flow_config.smoke_output_dir.parent)

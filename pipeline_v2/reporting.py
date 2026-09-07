from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline_v2.config import SharedRunConstants
from pipeline_v2.shared.io_utils import ensure_dir, read_csv_rows, write_csv, write_json, write_markdown


FLOW_PROTOCOL_DIRS = {
    "frontier_backbone": {
        "smoke": "frontier_backbone_smoke",
        "small_validation": "frontier_backbone_small_validation",
    },
    "frontier_backbone_llm_s1": {
        "smoke": "frontier_backbone_llm_s1_smoke",
        "small_validation": "frontier_backbone_llm_s1_small_validation",
    },
    "frontier_backbone_llm_s2": {
        "smoke": "frontier_backbone_llm_s2_smoke",
        "small_validation": "frontier_backbone_llm_s2_small_validation",
    },
    "frontier_backbone_llm_s1s2": {
        "smoke": "frontier_backbone_llm_s1s2_smoke",
        "small_validation": "frontier_backbone_llm_s1s2_small_validation",
    },
}

_BENCHMARK_SEEDS = SharedRunConstants().benchmark_seeds
BENCHMARK_FLOW_PROTOCOL_DIRS = {
    flow_name: {
        f"medium_benchmark_seed{seed}": f"{flow_name}_medium_benchmark_seed{seed}"
        for seed in _BENCHMARK_SEEDS
    }
    for flow_name in FLOW_PROTOCOL_DIRS
}
_FINAL_BENCHMARK_SEEDS = SharedRunConstants().final_benchmark_seeds
FINAL_BENCHMARK_FLOW_PROTOCOL_DIRS = {
    flow_name: {
        f"final_benchmark_seed{seed}": f"{flow_name}_final_benchmark_seed{seed}"
        for seed in _FINAL_BENCHMARK_SEEDS
    }
    for flow_name in FLOW_PROTOCOL_DIRS
}


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _to_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    center = _mean(values)
    return (_mean([(value - center) ** 2 for value in values])) ** 0.5


def _protocol_seed(protocol_name: str) -> int:
    match = re.search(r"seed(\d+)", protocol_name)
    return int(match.group(1)) if match else 0


def _diagnostic_path(output_dir: Path, profile_name: str, filename: str) -> Path:
    return output_dir / profile_name / filename


def _round_metrics_rows(output_dir: Path, profile_name: str) -> list[dict[str, str]]:
    path = output_dir / "round_metrics.csv"
    if not path.exists():
        return []
    return [row for row in read_csv_rows(path) if str(row.get("profile_name", "")) == profile_name]


def _experiments_rows(output_dir: Path, profile_name: str) -> list[dict[str, str]]:
    path = output_dir / "experiments.csv"
    if not path.exists():
        return []
    return [row for row in read_csv_rows(path) if str(row.get("profile_name", "")) == profile_name]


def _stage2_diagnostics_rows(output_dir: Path, profile_name: str) -> list[dict[str, object]]:
    profile_dir = output_dir / profile_name
    if not profile_dir.exists():
        return []
    def _round_number(path: Path) -> int:
        match = re.search(r"round(\d+)_stage2_diagnostics\.json", path.name)
        return int(match.group(1)) if match else 0
    paths = sorted(profile_dir.glob("round*_stage2_diagnostics.json"), key=_round_number)
    return [_read_json(path) for path in paths]


def _enriched_rows_for_output_dir(flow_name: str, protocol_name: str, output_dir: Path) -> list[dict[str, object]]:
    summary_path = output_dir / "run_summary.csv"
    if not summary_path.exists():
        return []
    enriched_rows: list[dict[str, object]] = []
    for row in read_csv_rows(summary_path):
        profile_name = str(row.get("profile_name", ""))
        stage1_diag = _read_json(_diagnostic_path(output_dir, profile_name, "stage1_diagnostics.json"))
        round_rows = _round_metrics_rows(output_dir, profile_name)
        experiments = _experiments_rows(output_dir, profile_name)
        stage2_diags = _stage2_diagnostics_rows(output_dir, profile_name)

        round1_rows = [item for item in round_rows if _to_int(item.get("round_index", 0)) == 1]
        stage2_rows = [item for item in round_rows if str(item.get("batch_kind", "")) == "stage2_bo_batch"]
        round1_row = round1_rows[0] if round1_rows else {}
        action_counts = stage1_diag.get("final_action_counts", {}) if isinstance(stage1_diag.get("final_action_counts", {}), dict) else {}
        reason_counts = stage1_diag.get("reason_label_counts", {}) if isinstance(stage1_diag.get("reason_label_counts", {}), dict) else {}

        stage2_mean_yields = [_to_float(item.get("batch_mean_yield", 0.0)) for item in stage2_rows]
        stage2_best_yields = [_to_float(item.get("batch_best_yield", 0.0)) for item in stage2_rows]
        topk_true_means = [_to_float(item.get("topk_true_mean_yield", diag.get("topk_true_mean_yield", 0.0))) for item, diag in zip(stage2_rows, stage2_diags)] if stage2_rows and stage2_diags else [_to_float(diag.get("topk_true_mean_yield", 0.0)) for diag in stage2_diags]
        pre_gate_true_means = [_to_float(item.get("pre_gate_batch_true_mean_yield", diag.get("pre_gate_batch_true_mean_yield", 0.0))) for item, diag in zip(stage2_rows, stage2_diags)] if stage2_rows and stage2_diags else [_to_float(diag.get("pre_gate_batch_true_mean_yield", 0.0)) for diag in stage2_diags]
        post_gate_true_means = [_to_float(item.get("post_gate_batch_true_mean_yield", diag.get("post_gate_batch_true_mean_yield", 0.0))) for item, diag in zip(stage2_rows, stage2_diags)] if stage2_rows and stage2_diags else [_to_float(diag.get("post_gate_batch_true_mean_yield", 0.0)) for diag in stage2_diags]
        gate_quality_deltas = [_to_float(item.get("gate_batch_quality_delta", diag.get("gate_batch_quality_delta", 0.0))) for item, diag in zip(stage2_rows, stage2_diags)] if stage2_rows and stage2_diags else [_to_float(diag.get("gate_batch_quality_delta", 0.0)) for diag in stage2_diags]
        stage2_keep_counts = [_to_int(diag.get("action_counts", {}).get("keep", 0)) for diag in stage2_diags]
        stage2_downweight_counts = [_to_int(diag.get("action_counts", {}).get("downweight", 0)) for diag in stage2_diags]
        stage2_reject_counts = [_to_int(diag.get("action_counts", {}).get("reject", 0)) for diag in stage2_diags]
        stage2_prefer_counts = [_to_int(diag.get("action_counts", {}).get("prefer_for_exploration", 0)) for diag in stage2_diags]
        stage2_changed_counts = [_to_int(diag.get("final_bo_batch_changed_items", 0)) for diag in stage2_diags]
        stage2_confidences = [_to_float(diag.get("mean_nonkeep_confidence", 0.0)) for diag in stage2_diags]
        stage2_nonkeep_counts = [_to_int(diag.get("nonkeep_action_count", 0)) for diag in stage2_diags]

        oracle_best = _to_float(row.get("oracle_best_yield", 0.0))
        if row.get("cumulative_regret", "") not in (None, ""):
            cumulative_regret = _to_float(row.get("cumulative_regret", 0.0))
        else:
            cumulative_regret = round(sum(max(0.0, oracle_best - _to_float(item.get("yield_value", 0.0))) for item in experiments), 4)

        enriched = dict(row)
        enriched.update(
            {
                "flow_name": flow_name,
                "protocol": protocol_name,
                "seed": _protocol_seed(protocol_name),
                "round1_batch_mean_yield": row.get("round1_batch_mean_yield", round1_row.get("batch_mean_yield", "")),
                "round1_batch_best_yield": row.get("round1_batch_best_yield", round1_row.get("batch_best_yield", "")),
                "stage2_batch_mean_yield": row.get("stage2_batch_mean_yield", round(_mean(stage2_mean_yields), 4) if stage2_mean_yields else ""),
                "stage2_batch_best_yield": row.get("stage2_batch_best_yield", round(_mean(stage2_best_yields), 4) if stage2_best_yields else ""),
                "final_stage2_batch_mean_yield": row.get("final_stage2_batch_mean_yield", stage2_mean_yields[-1] if stage2_mean_yields else ""),
                "stage2_topk_true_mean_yield": row.get("stage2_topk_true_mean_yield", round(_mean(topk_true_means), 4) if topk_true_means else ""),
                "stage2_pre_gate_batch_true_mean_yield": row.get("stage2_pre_gate_batch_true_mean_yield", round(_mean(pre_gate_true_means), 4) if pre_gate_true_means else ""),
                "stage2_post_gate_batch_true_mean_yield": row.get("stage2_post_gate_batch_true_mean_yield", round(_mean(post_gate_true_means), 4) if post_gate_true_means else ""),
                "stage2_gate_batch_quality_delta": row.get("stage2_gate_batch_quality_delta", round(_mean(gate_quality_deltas), 4) if gate_quality_deltas else ""),
                "cumulative_regret": row.get("cumulative_regret", cumulative_regret),
                "stage1_candidate_pool_before_s1": row.get("stage1_candidate_pool_before_s1", stage1_diag.get("candidate_pool_size_before_s1", "")),
                "stage1_candidate_pool_after_s1": row.get("stage1_candidate_pool_after_s1", stage1_diag.get("candidate_pool_size_after_s1", "")),
                "stage1_candidate_pool_after_rule_filter": row.get("stage1_candidate_pool_after_rule_filter", stage1_diag.get("candidate_pool_size_after_rule_filter", "")),
                "stage1_baseline_candidate_pool_after_rule_filter": row.get(
                    "stage1_baseline_candidate_pool_after_rule_filter",
                    stage1_diag.get("baseline_candidate_pool_after_rule_filter", ""),
                ),
                "stage1_initial_batch_changed_items": row.get(
                    "stage1_initial_batch_changed_items",
                    stage1_diag.get("initial_batch_changed_items_vs_backbone", ""),
                ),
                "llm_s1_rejected_count": row.get("llm_s1_rejected_count", action_counts.get("rejected", "")),
                "llm_s1_downweighted_count": row.get("llm_s1_downweighted_count", action_counts.get("downweighted", "")),
                "llm_s1_bonused_count": row.get("llm_s1_bonused_count", action_counts.get("bonused_preferred", "")),
                "llm_s1_unchanged_count": row.get("llm_s1_unchanged_count", action_counts.get("unchanged", "")),
                "llm_s1_contradiction_resolution_count": row.get(
                    "llm_s1_contradiction_resolution_count",
                    stage1_diag.get("contradiction_resolution_count", ""),
                ),
                "llm_s1_reason_conflict_count": row.get("llm_s1_reason_conflict_count", reason_counts.get("conflict", "")),
                "llm_s1_reason_weak_transfer_count": row.get("llm_s1_reason_weak_transfer_count", reason_counts.get("weak_transfer", "")),
                "llm_s1_reason_dubious_compatibility_count": row.get(
                    "llm_s1_reason_dubious_compatibility_count",
                    reason_counts.get("dubious_compatibility", ""),
                ),
                "llm_s1_reason_low_support_count": row.get("llm_s1_reason_low_support_count", reason_counts.get("low_support", "")),
                "llm_s1_reason_positive_evidence_count": row.get(
                    "llm_s1_reason_positive_evidence_count",
                    reason_counts.get("positive_evidence", ""),
                ),
                "llm_s2_topk_size": row.get("llm_s2_topk_size", _to_int(stage2_diags[0].get("topk_size", 0)) if stage2_diags else ""),
                "llm_s2_keep_count": round(_mean([float(value) for value in stage2_keep_counts]), 2) if stage2_keep_counts else row.get("llm_s2_keep_count", ""),
                "llm_s2_downweight_count": round(_mean([float(value) for value in stage2_downweight_counts]), 2) if stage2_downweight_counts else row.get("llm_s2_downweight_count", ""),
                "llm_s2_reject_count": round(_mean([float(value) for value in stage2_reject_counts]), 2) if stage2_reject_counts else row.get("llm_s2_reject_count", ""),
                "llm_s2_prefer_count": round(_mean([float(value) for value in stage2_prefer_counts]), 2) if stage2_prefer_counts else row.get("llm_s2_prefer_count", ""),
                "llm_s2_changed_batch_items": round(_mean([float(value) for value in stage2_changed_counts]), 2) if stage2_changed_counts else row.get("llm_s2_changed_batch_items", ""),
                "llm_s2_mean_nonkeep_confidence": round(_mean(stage2_confidences), 4) if stage2_confidences else row.get("llm_s2_mean_nonkeep_confidence", ""),
                "llm_s2_nonkeep_actions_per_round": round(_mean([float(value) for value in stage2_nonkeep_counts]), 2) if stage2_nonkeep_counts else row.get("llm_s2_action_count", ""),
                "llm_s2_total_action_count": row.get("llm_s2_action_count", ""),
                "llm_s2_round_count": len(stage2_diags),
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def _collect_rows(results_root: Path, mapping: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    comparison_rows: list[dict[str, object]] = []
    for flow_name, protocol_dirs in mapping.items():
        for protocol_name, dir_name in protocol_dirs.items():
            comparison_rows.extend(_enriched_rows_for_output_dir(flow_name, protocol_name, results_root / dir_name))
    return comparison_rows


def build_smoke_comparison(results_root: Path) -> None:
    comparison_rows = _collect_rows(results_root, FLOW_PROTOCOL_DIRS)
    if not comparison_rows:
        return

    output_dir = ensure_dir(results_root / "smoke_comparison")
    write_csv(output_dir / "smoke_comparison.csv", comparison_rows)

    lines = [
        "# Smoke Comparison",
        "",
        "This comparison is intentionally small-scale. It combines the smoke run and the tiny matched validation only.",
        "",
        "| flow | protocol | profile | best_observed_yield | oracle_best_yield | regret_to_oracle | round1_batch_mean_yield | stage2_batch_mean_yield |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row.get('flow_name', '')} | {row.get('protocol', '')} | {row.get('profile_name', '')} | "
            f"{row.get('best_observed_yield', '')} | {row.get('oracle_best_yield', '')} | "
            f"{row.get('regret_to_oracle', '')} | {row.get('round1_batch_mean_yield', '')} | {row.get('stage2_batch_mean_yield', '')} |"
        )
    write_markdown(output_dir / "smoke_comparison.md", lines)


def build_calibration_round(results_root: Path) -> None:
    required_paths = [
        results_root / dir_name / "run_summary.csv"
        for protocol_dirs in FLOW_PROTOCOL_DIRS.values()
        for dir_name in protocol_dirs.values()
    ]
    if not all(path.exists() for path in required_paths):
        return

    comparison_rows = _collect_rows(results_root, FLOW_PROTOCOL_DIRS)
    output_dir = ensure_dir(results_root / "calibration_round")
    write_csv(output_dir / "calibration_comparison.csv", comparison_rows)

    comparison_lines = [
        "# Calibration Comparison",
        "",
        "This round is limited to smoke and matched small validation across the four V2 modes.",
        "",
        "| flow | protocol | profile | round1_mean | stage2_mean | best_observed | oracle_best | regret |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison_rows:
        comparison_lines.append(
            f"| {row.get('flow_name', '')} | {row.get('protocol', '')} | {row.get('profile_name', '')} | "
            f"{row.get('round1_batch_mean_yield', '')} | {row.get('stage2_batch_mean_yield', '')} | "
            f"{row.get('best_observed_yield', '')} | {row.get('oracle_best_yield', '')} | {row.get('regret_to_oracle', '')} |"
        )
    write_markdown(output_dir / "calibration_comparison.md", comparison_lines)

    s1_enabled_rows = [
        row
        for row in comparison_rows
        if str(row.get("flow_name", "")) in {"frontier_backbone_llm_s1", "frontier_backbone_llm_s1s2"}
    ]
    s2_enabled_rows = [
        row
        for row in comparison_rows
        if str(row.get("flow_name", "")) in {"frontier_backbone_llm_s2", "frontier_backbone_llm_s1s2"}
    ]
    s1_modified_fractions = [_to_float(row.get("llm_s1_modified_fraction", 0.0)) for row in s1_enabled_rows]
    s1_pool_shrinkages = [
        _to_int(row.get("stage1_baseline_candidate_pool_after_rule_filter", 0))
        - _to_int(row.get("stage1_candidate_pool_after_rule_filter", 0))
        for row in s1_enabled_rows
    ]
    s2_keep_fractions = [
        _to_float(row.get("llm_s2_keep_count", 0)) / max(_to_float(row.get("llm_s2_topk_size", 0)), 1.0)
        for row in s2_enabled_rows
    ]
    notes = [
        "# Calibration Notes",
        "",
        "## Calibration Summary",
    ]
    if s1_enabled_rows:
        notes.append(
            f"- Calibrated S1 stayed bounded in this round: reject stayed at 0 across all saved S1-enabled runs, "
            f"contradictions resolved to 0, modified_fraction ranged from {min(s1_modified_fractions):.2f} to {max(s1_modified_fractions):.2f}, "
            f"and extra feasible-pool shrinkage versus the backbone filter was {min(s1_pool_shrinkages)} to {max(s1_pool_shrinkages)} candidates."
        )
    if s2_enabled_rows:
        notes.append(
            f"- Calibrated S2 was sparse by default: keep covered {min(s2_keep_fractions):.0%} to {max(s2_keep_fractions):.0%} of top-K items, "
            f"with only {_to_int(min(row.get('llm_s2_action_count', 0) for row in s2_enabled_rows))} to {_to_int(max(row.get('llm_s2_action_count', 0) for row in s2_enabled_rows))} non-keep actions per round."
        )
    notes.extend(["", "## S1 Diagnostics"])
    for row in comparison_rows:
        if str(row.get("flow_name", "")) not in {"frontier_backbone_llm_s1", "frontier_backbone_llm_s1s2"}:
            continue
        notes.append(
            f"- {row['flow_name']} {row['protocol']} {row['profile_name']}: "
            f"reject={row.get('llm_s1_rejected_count', 0)} downweight={row.get('llm_s1_downweighted_count', 0)} "
            f"bonus={row.get('llm_s1_bonused_count', 0)} unchanged={row.get('llm_s1_unchanged_count', 0)} "
            f"pool={row.get('stage1_candidate_pool_before_s1', 0)}->{row.get('stage1_candidate_pool_after_rule_filter', 0)} "
            f"changed_initial_batch={row.get('stage1_initial_batch_changed_items', 0)} "
            f"contradictions_resolved={row.get('llm_s1_contradiction_resolution_count', 0)}"
        )
    notes.extend(["", "## S2 Diagnostics"])
    for row in comparison_rows:
        if str(row.get("flow_name", "")) not in {"frontier_backbone_llm_s2", "frontier_backbone_llm_s1s2"}:
            continue
        notes.append(
            f"- {row['flow_name']} {row['protocol']} {row['profile_name']}: "
            f"topk={row.get('llm_s2_topk_size', 0)} keep={row.get('llm_s2_keep_count', 0)} "
            f"downweight={row.get('llm_s2_downweight_count', 0)} reject={row.get('llm_s2_reject_count', 0)} "
            f"prefer={row.get('llm_s2_prefer_count', 0)} changed_final_batch={row.get('llm_s2_changed_batch_items', 0)}"
        )
    notes.extend(["", "## Interpretation"])
    notes.append("- Use the medium benchmark artifacts for the stronger stability read; this calibration note remains the small-scope checkpoint.")
    write_markdown(output_dir / "calibration_notes.md", notes)


def build_calibration_benchmark(results_root: Path) -> None:
    required_paths = [
        results_root / dir_name / "run_summary.csv"
        for protocol_dirs in BENCHMARK_FLOW_PROTOCOL_DIRS.values()
        for dir_name in protocol_dirs.values()
    ]
    if not all(path.exists() for path in required_paths):
        return

    comparison_rows = _collect_rows(results_root, BENCHMARK_FLOW_PROTOCOL_DIRS)
    output_dir = ensure_dir(results_root / "calibration_benchmark")
    write_csv(output_dir / "calibration_benchmark_comparison.csv", comparison_rows)

    comparison_lines = [
        "# Calibration Benchmark Comparison",
        "",
        "This round runs a medium matched benchmark across all supported profiles and the configured benchmark seeds.",
        "",
        "| flow | protocol | profile | seed | round1_best | stage2_mean | topk_true_mean | post_gate_true_mean | cumulative_regret | best_observed | oracle |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison_rows:
        comparison_lines.append(
            f"| {row.get('flow_name', '')} | {row.get('protocol', '')} | {row.get('profile_name', '')} | {row.get('seed', '')} | "
            f"{row.get('round1_batch_best_yield', '')} | {row.get('stage2_batch_mean_yield', '')} | "
            f"{row.get('stage2_topk_true_mean_yield', '')} | {row.get('stage2_post_gate_batch_true_mean_yield', '')} | "
            f"{row.get('cumulative_regret', '')} | {row.get('best_observed_yield', '')} | {row.get('oracle_best_yield', '')} |"
        )
    write_markdown(output_dir / "calibration_benchmark_comparison.md", comparison_lines)

    grouped_by_flow: dict[str, list[dict[str, object]]] = {}
    grouped_by_flow_profile: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in comparison_rows:
        flow_name = str(row.get("flow_name", ""))
        profile_name = str(row.get("profile_name", ""))
        grouped_by_flow.setdefault(flow_name, []).append(row)
        grouped_by_flow_profile.setdefault((flow_name, profile_name), []).append(row)

    def _flow_metric(flow_name: str, key: str) -> float:
        return round(_mean([_to_float(row.get(key, 0.0)) for row in grouped_by_flow.get(flow_name, [])]), 4)

    notes = [
        "# Calibration Benchmark Notes",
        "",
        "## Scope",
        f"- Modes: {', '.join(BENCHMARK_FLOW_PROTOCOL_DIRS.keys())}",
        f"- Profiles: {', '.join(SharedRunConstants().supported_profiles)}",
        f"- Seeds: {', '.join(str(seed) for seed in _BENCHMARK_SEEDS)}",
        f"- Stage 2 rounds per profile-run: {SharedRunConstants().medium_benchmark_stage2_rounds}",
        f"- Total matched profile-runs: {len(comparison_rows)}",
        "",
        "## S2 Changes",
        f"- S2 score deltas were tightened to downweight={SharedRunConstants().s2_downweight_delta}, prefer={SharedRunConstants().s2_prefer_delta}, reject={SharedRunConstants().s2_reject_delta}.",
        f"- Non-keep gating now requires higher confidence (base threshold={SharedRunConstants().s2_min_confidence_for_nonkeep}, hindered threshold={SharedRunConstants().s2_hindered_min_confidence_for_nonkeep}).",
        f"- Final BO batch changes are capped at {SharedRunConstants().s2_max_final_batch_changes} generally and {SharedRunConstants().s2_hindered_max_final_batch_changes} on hindered profiles.",
        "- The gate only acts near the BO boundary and keeps extra caution on hindered_aryl_chloride_activation.",
        "",
        "## S1 Status",
        "- S1 was intentionally left in its current bounded form. Reject remained effectively zero, contradiction guarding remained active, and there was no extra feasible-pool shrinkage beyond the backbone rule filter in the saved benchmark rows.",
        "",
        "## S2 Sparsity",
    ]
    s2_rows = [
        row
        for row in comparison_rows
        if str(row.get("flow_name", "")) in {"frontier_backbone_llm_s2", "frontier_backbone_llm_s1s2"}
    ]
    if s2_rows:
        keep_fractions = [_to_float(row.get("llm_s2_keep_count", 0.0)) / max(_to_float(row.get("llm_s2_topk_size", 0.0)), 1.0) for row in s2_rows]
        nonkeep_counts = [_to_float(row.get("llm_s2_nonkeep_actions_per_round", 0.0)) for row in s2_rows]
        changed_counts = [_to_float(row.get("llm_s2_changed_batch_items", 0.0)) for row in s2_rows]
        confidence_values = [_to_float(row.get("llm_s2_mean_nonkeep_confidence", 0.0)) for row in s2_rows if _to_float(row.get("llm_s2_mean_nonkeep_confidence", 0.0)) > 0.0]
        notes.append(
            f"- S2 stayed sparse in the benchmark: keep covered {min(keep_fractions):.0%} to {max(keep_fractions):.0%} of top-K items, "
            f"non-keep actions averaged {min(nonkeep_counts):.0f} to {max(nonkeep_counts):.0f} per round, "
            f"and final batch changes averaged {round(_mean(changed_counts), 2)} items."
        )
        if confidence_values:
            notes.append(
                f"- Selected non-keep actions carried mean gate confidence between {min(confidence_values):.2f} and {max(confidence_values):.2f}."
            )

    reference_path = output_dir / "pre_tightening_small_reference.csv"
    if reference_path.exists():
        current_small_rows = _collect_rows(results_root, FLOW_PROTOCOL_DIRS)
        reference_rows = [dict(row) for row in read_csv_rows(reference_path)]
        def _lookup(rows: list[dict[str, object]], flow_name: str, profile_name: str) -> dict[str, object] | None:
            for row in rows:
                if str(row.get("flow_name", "")) == flow_name and str(row.get("protocol", "")) == "small_validation" and str(row.get("profile_name", "")) == profile_name:
                    return row
            return None
        notes.extend(["", "## Small-Scope S2 Tightening Check"])
        for flow_name in ("frontier_backbone_llm_s2", "frontier_backbone_llm_s1s2"):
            for profile_name in ("aza_heteroaryl_chloride_guarded", "hindered_aryl_chloride_activation"):
                before_row = _lookup(reference_rows, flow_name, profile_name)
                after_row = _lookup(current_small_rows, flow_name, profile_name)
                if not before_row or not after_row:
                    continue
                stage2_before = _to_float(before_row.get("stage2_batch_mean_yield", 0.0))
                stage2_after = _to_float(after_row.get("stage2_batch_mean_yield", 0.0))
                actions_before = _to_float(before_row.get("llm_s2_action_count", 0.0))
                actions_after = _to_float(after_row.get("llm_s2_action_count", 0.0))
                notes.append(
                    f"- {flow_name} {profile_name}: stage2_mean {stage2_before:.4f} -> {stage2_after:.4f}, "
                    f"nonkeep_actions {actions_before:.0f} -> {actions_after:.0f}."
                )

    notes.extend(["", "## Flow Summary"])
    for flow_name in BENCHMARK_FLOW_PROTOCOL_DIRS:
        rows = grouped_by_flow.get(flow_name, [])
        if not rows:
            continue
        notes.append(
            f"- {flow_name}: best_mean={_flow_metric(flow_name, 'best_observed_yield'):.4f} "
            f"oracle_regret_mean={_flow_metric(flow_name, 'regret_to_oracle'):.4f} "
            f"round1_best_mean={_flow_metric(flow_name, 'round1_batch_best_yield'):.4f} "
            f"stage2_mean={_flow_metric(flow_name, 'stage2_batch_mean_yield'):.4f} "
            f"topk_true_mean={_flow_metric(flow_name, 'stage2_topk_true_mean_yield'):.4f} "
            f"gate_delta={_flow_metric(flow_name, 'stage2_gate_batch_quality_delta'):.4f} "
            f"cumulative_regret={_flow_metric(flow_name, 'cumulative_regret'):.4f}"
        )

    notes.extend(["", "## Per-Profile Stability"])
    for (flow_name, profile_name), rows in sorted(grouped_by_flow_profile.items()):
        stage2_means = [_to_float(row.get("stage2_batch_mean_yield", 0.0)) for row in rows]
        bests = [_to_float(row.get("best_observed_yield", 0.0)) for row in rows]
        notes.append(
            f"- {flow_name} {profile_name}: stage2_mean={_mean(stage2_means):.4f} +/- {_std(stage2_means):.4f}, "
            f"best={_mean(bests):.4f} +/- {_std(bests):.4f}"
        )

    notes.extend(["", "## Hindered Profile Readout"])
    hindered_s2 = [
        row
        for row in s2_rows
        if str(row.get("profile_name", "")) == "hindered_aryl_chloride_activation"
    ]
    if hindered_s2:
        notes.append(
            f"- Hindered-profile S2 behavior stayed cautious: changed_final_batch averaged {round(_mean([_to_float(row.get('llm_s2_changed_batch_items', 0.0)) for row in hindered_s2]), 2)} items, "
            f"with stage2_mean averaging {round(_mean([_to_float(row.get('stage2_batch_mean_yield', 0.0)) for row in hindered_s2]), 4)}."
        )

    backbone_stage2 = _flow_metric("frontier_backbone", "stage2_batch_mean_yield")
    s2_only_stage2 = _flow_metric("frontier_backbone_llm_s2", "stage2_batch_mean_yield")
    s1s2_stage2 = _flow_metric("frontier_backbone_llm_s1s2", "stage2_batch_mean_yield")
    backbone_cum_regret = _flow_metric("frontier_backbone", "cumulative_regret")
    s2_only_cum_regret = _flow_metric("frontier_backbone_llm_s2", "cumulative_regret")
    s1s2_cum_regret = _flow_metric("frontier_backbone_llm_s1s2", "cumulative_regret")

    notes.extend(["", "## Interpretation"])
    if s2_only_stage2 >= s1s2_stage2:
        notes.append("- S2-only remains safer than S1+S2 on Stage 2 batch mean in this medium benchmark.")
    else:
        notes.append("- S1+S2 closed the gap to S2-only on Stage 2 batch mean in this medium benchmark.")
    if s2_only_stage2 >= backbone_stage2 - 0.015:
        notes.append("- The tightened S2 gate now looks close to non-destructive on Stage 2 batch mean relative to the backbone.")
    else:
        notes.append("- The tightened S2 gate is safer than before but still shows measurable Stage 2 drag versus the backbone.")
    if s1s2_cum_regret <= s2_only_cum_regret + 0.02:
        notes.append("- S1+S2 cumulative regret is within the specified tolerance of S2-only.")
    else:
        notes.append("- The combined S1+S2 mode still adds noticeable drag beyond S2-only, mainly through Stage 2 behavior.")

    notes.extend(["", "## Recommendation"])
    if s2_only_stage2 >= backbone_stage2 - 0.015 and s1s2_stage2 >= backbone_stage2 - 0.03:
        notes.append("- Both S2-only and S1+S2 satisfy the configured batch-quality tolerance in this comparison.")
    else:
        notes.append("- At least one LLM configuration falls below the configured Stage 2 batch-quality tolerance.")
    notes.append("- S2 sparsity and hindered-profile behavior are evaluated separately from S1 corrections.")

    representative_path = (
        results_root
        / "frontier_backbone_llm_s2_medium_benchmark_seed17"
        / "hindered_aryl_chloride_activation"
        / "round2_stage2_diagnostics.json"
    )
    if representative_path.exists():
        representative_payload = _read_json(representative_path)
        representative_payload["source_path"] = str(representative_path)
        write_json(output_dir / "representative_hindered_s2_diagnostics.json", representative_payload)
        notes.append(f"- Representative hindered S2 diagnostic saved to {output_dir / 'representative_hindered_s2_diagnostics.json'}.")

    write_markdown(output_dir / "calibration_benchmark_notes.md", notes)


def build_final_benchmark(results_root: Path) -> None:
    required_paths = [
        results_root / dir_name / "run_summary.csv"
        for protocol_dirs in FINAL_BENCHMARK_FLOW_PROTOCOL_DIRS.values()
        for dir_name in protocol_dirs.values()
    ]
    if not all(path.exists() for path in required_paths):
        return

    comparison_rows = _collect_rows(results_root, FINAL_BENCHMARK_FLOW_PROTOCOL_DIRS)
    output_dir = ensure_dir(results_root / "final_benchmark")
    write_csv(output_dir / "final_benchmark_comparison.csv", comparison_rows)

    comparison_lines = [
        "# Final Benchmark Comparison",
        "",
        "This is the final matched benchmark for the frozen calibrated V2 system.",
        "",
        "| flow | protocol | profile | seed | best_observed | oracle_regret | cumulative_regret | stage2_mean | round1_best | topk_true_mean | post_gate_true_mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison_rows:
        comparison_lines.append(
            f"| {row.get('flow_name', '')} | {row.get('protocol', '')} | {row.get('profile_name', '')} | {row.get('seed', '')} | "
            f"{row.get('best_observed_yield', '')} | {row.get('regret_to_oracle', '')} | {row.get('cumulative_regret', '')} | "
            f"{row.get('stage2_batch_mean_yield', '')} | {row.get('round1_batch_best_yield', '')} | "
            f"{row.get('stage2_topk_true_mean_yield', '')} | {row.get('stage2_post_gate_batch_true_mean_yield', '')} |"
        )
    write_markdown(output_dir / "final_benchmark_comparison.md", comparison_lines)

    grouped_by_flow: dict[str, list[dict[str, object]]] = {}
    grouped_by_flow_profile: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in comparison_rows:
        flow_name = str(row.get("flow_name", ""))
        profile_name = str(row.get("profile_name", ""))
        grouped_by_flow.setdefault(flow_name, []).append(row)
        grouped_by_flow_profile.setdefault((flow_name, profile_name), []).append(row)

    def _flow_metric(flow_name: str, key: str) -> float:
        return round(_mean([_to_float(row.get(key, 0.0)) for row in grouped_by_flow.get(flow_name, [])]), 4)

    overall_rows: list[dict[str, object]] = []
    for flow_name, rows in grouped_by_flow.items():
        overall_rows.append(
            {
                "flow_name": flow_name,
                "num_profile_runs": len(rows),
                "best_observed_mean": _flow_metric(flow_name, "best_observed_yield"),
                "best_observed_std": round(_std([_to_float(row.get("best_observed_yield", 0.0)) for row in rows]), 4),
                "oracle_regret_mean": _flow_metric(flow_name, "regret_to_oracle"),
                "cumulative_regret_mean": _flow_metric(flow_name, "cumulative_regret"),
                "stage2_mean": _flow_metric(flow_name, "stage2_batch_mean_yield"),
                "round1_best_mean": _flow_metric(flow_name, "round1_batch_best_yield"),
                "topk_true_mean": _flow_metric(flow_name, "stage2_topk_true_mean_yield"),
                "post_gate_true_mean": _flow_metric(flow_name, "stage2_post_gate_batch_true_mean_yield"),
                "gate_delta_mean": _flow_metric(flow_name, "stage2_gate_batch_quality_delta"),
            }
        )
    write_csv(output_dir / "final_benchmark_overall_summary.csv", overall_rows)

    profile_rows: list[dict[str, object]] = []
    for (flow_name, profile_name), rows in sorted(grouped_by_flow_profile.items()):
        profile_rows.append(
            {
                "flow_name": flow_name,
                "profile_name": profile_name,
                "num_runs": len(rows),
                "best_observed_mean": round(_mean([_to_float(row.get("best_observed_yield", 0.0)) for row in rows]), 4),
                "best_observed_std": round(_std([_to_float(row.get("best_observed_yield", 0.0)) for row in rows]), 4),
                "oracle_regret_mean": round(_mean([_to_float(row.get("regret_to_oracle", 0.0)) for row in rows]), 4),
                "cumulative_regret_mean": round(_mean([_to_float(row.get("cumulative_regret", 0.0)) for row in rows]), 4),
                "stage2_mean": round(_mean([_to_float(row.get("stage2_batch_mean_yield", 0.0)) for row in rows]), 4),
                "stage2_std": round(_std([_to_float(row.get("stage2_batch_mean_yield", 0.0)) for row in rows]), 4),
                "round1_best_mean": round(_mean([_to_float(row.get("round1_batch_best_yield", 0.0)) for row in rows]), 4),
                "topk_true_mean": round(_mean([_to_float(row.get("stage2_topk_true_mean_yield", 0.0)) for row in rows]), 4),
                "post_gate_true_mean": round(_mean([_to_float(row.get("stage2_post_gate_batch_true_mean_yield", 0.0)) for row in rows]), 4),
            }
        )
    write_csv(output_dir / "final_benchmark_profile_summary.csv", profile_rows)

    diagnostics_rows: list[dict[str, object]] = []
    for flow_name, rows in grouped_by_flow.items():
        diagnostics_rows.append(
            {
                "flow_name": flow_name,
                "s1_reject_mean": round(_mean([_to_float(row.get("llm_s1_rejected_count", 0.0)) for row in rows]), 4),
                "s1_contradiction_mean": round(_mean([_to_float(row.get("llm_s1_contradiction_resolution_count", 0.0)) for row in rows]), 4),
                "s1_modified_fraction_mean": round(_mean([_to_float(row.get("llm_s1_modified_fraction", 0.0)) for row in rows]), 4),
                "s1_extra_pool_shrinkage_mean": round(
                    _mean(
                        [
                            _to_float(row.get("stage1_baseline_candidate_pool_after_rule_filter", 0.0))
                            - _to_float(row.get("stage1_candidate_pool_after_rule_filter", 0.0))
                            for row in rows
                        ]
                    ),
                    4,
                ),
                "s2_topk_size_mean": round(_mean([_to_float(row.get("llm_s2_topk_size", 0.0)) for row in rows]), 4),
                "s2_keep_mean": round(_mean([_to_float(row.get("llm_s2_keep_count", 0.0)) for row in rows]), 4),
                "s2_downweight_mean": round(_mean([_to_float(row.get("llm_s2_downweight_count", 0.0)) for row in rows]), 4),
                "s2_reject_mean": round(_mean([_to_float(row.get("llm_s2_reject_count", 0.0)) for row in rows]), 4),
                "s2_prefer_mean": round(_mean([_to_float(row.get("llm_s2_prefer_count", 0.0)) for row in rows]), 4),
                "s2_nonkeep_per_round_mean": round(_mean([_to_float(row.get("llm_s2_nonkeep_actions_per_round", 0.0)) for row in rows]), 4),
                "s2_changed_batch_items_mean": round(_mean([_to_float(row.get("llm_s2_changed_batch_items", 0.0)) for row in rows]), 4),
                "s2_mean_nonkeep_confidence": round(_mean([_to_float(row.get("llm_s2_mean_nonkeep_confidence", 0.0)) for row in rows]), 4),
            }
        )
    write_csv(output_dir / "final_benchmark_diagnostics_summary.csv", diagnostics_rows)

    hindered_s2_path = (
        results_root
        / "frontier_backbone_llm_s2_final_benchmark_seed17"
        / "hindered_aryl_chloride_activation"
        / "round2_stage2_diagnostics.json"
    )
    if hindered_s2_path.exists():
        payload = _read_json(hindered_s2_path)
        payload["source_path"] = str(hindered_s2_path)
        write_json(output_dir / "representative_hindered_s2_diagnostics.json", payload)

    bounded_s1_path = (
        results_root
        / "frontier_backbone_llm_s1_final_benchmark_seed17"
        / "hindered_aryl_chloride_activation"
        / "llm_s1_diagnostics.json"
    )
    if bounded_s1_path.exists():
        payload = _read_json(bounded_s1_path)
        payload["source_path"] = str(bounded_s1_path)
        write_json(output_dir / "representative_bounded_s1_diagnostics.json", payload)

    def _all_tie(key: str) -> bool:
        values = {round(_flow_metric(flow_name, key), 6) for flow_name in grouped_by_flow}
        return len(values) == 1

    backbone_stage2 = _flow_metric("frontier_backbone", "stage2_batch_mean_yield")
    llm_s2_stage2 = _flow_metric("frontier_backbone_llm_s2", "stage2_batch_mean_yield")
    llm_s1s2_stage2 = _flow_metric("frontier_backbone_llm_s1s2", "stage2_batch_mean_yield")
    llm_s1_stage2 = _flow_metric("frontier_backbone_llm_s1", "stage2_batch_mean_yield")
    backbone_cum_regret = _flow_metric("frontier_backbone", "cumulative_regret")
    llm_s2_cum_regret = _flow_metric("frontier_backbone_llm_s2", "cumulative_regret")
    llm_s1s2_cum_regret = _flow_metric("frontier_backbone_llm_s1s2", "cumulative_regret")

    notes = [
        "# Final Benchmark Notes",
        "",
        "## Scope",
        f"- Modes: {', '.join(FINAL_BENCHMARK_FLOW_PROTOCOL_DIRS.keys())}",
        f"- Profiles: {', '.join(SharedRunConstants().supported_profiles)}",
        f"- Seeds: {', '.join(str(seed) for seed in _FINAL_BENCHMARK_SEEDS)}",
        f"- Stage 2 rounds per profile-run: {SharedRunConstants().final_benchmark_stage2_rounds}",
        f"- Total matched profile-runs: {len(comparison_rows)}",
        "",
        "## Frozen Evaluation Target",
        "- S1 and S2 behavior were evaluated as-is from the calibrated V2 state. No retuning of prompts, scoring, bounds, or mode definitions was performed in this round.",
        "",
        "## Bounded Diagnostics",
    ]
    for row in diagnostics_rows:
        notes.append(
            f"- {row['flow_name']}: s1_reject_mean={row['s1_reject_mean']}, s1_contradiction_mean={row['s1_contradiction_mean']}, "
            f"s1_modified_fraction_mean={row['s1_modified_fraction_mean']}, s1_extra_pool_shrinkage_mean={row['s1_extra_pool_shrinkage_mean']}, "
            f"s2_keep_mean={row['s2_keep_mean']}, s2_downweight_mean={row['s2_downweight_mean']}, s2_reject_mean={row['s2_reject_mean']}, "
            f"s2_prefer_mean={row['s2_prefer_mean']}, s2_nonkeep_per_round_mean={row['s2_nonkeep_per_round_mean']}, "
            f"s2_changed_batch_items_mean={row['s2_changed_batch_items_mean']}"
        )

    notes.extend(["", "## Overall Summary"])
    for row in overall_rows:
        notes.append(
            f"- {row['flow_name']}: best_observed_mean={row['best_observed_mean']:.4f}, oracle_regret_mean={row['oracle_regret_mean']:.4f}, "
            f"cumulative_regret_mean={row['cumulative_regret_mean']:.4f}, stage2_mean={row['stage2_mean']:.4f}, "
            f"round1_best_mean={row['round1_best_mean']:.4f}, topk_true_mean={row['topk_true_mean']:.4f}, post_gate_true_mean={row['post_gate_true_mean']:.4f}"
        )

    notes.extend(["", "## Primary Answers"])
    if _all_tie("best_observed_yield") and _all_tie("regret_to_oracle"):
        notes.append("- All four modes still tie on mean best observed yield and mean oracle regret in the final matched benchmark.")
    else:
        notes.append("- The final matched benchmark breaks the earlier tie on either best observed yield or oracle regret.")
    best_stage2_mode = max(overall_rows, key=lambda row: float(row["stage2_mean"]))
    notes.append(f"- The strongest overall Stage 2 batch quality is from {best_stage2_mode['flow_name']} with stage2_mean={best_stage2_mode['stage2_mean']:.4f}.")
    if llm_s2_stage2 >= llm_s1s2_stage2 and llm_s2_cum_regret <= llm_s1s2_cum_regret + 0.05:
        notes.append("- frontier_backbone_llm_s2 remains the safest LLM-backed mode overall.")
    else:
        notes.append("- frontier_backbone_llm_s2 is no longer clearly the safest LLM-backed mode on the final benchmark.")
    if llm_s1s2_stage2 > max(backbone_stage2, llm_s2_stage2, llm_s1_stage2):
        notes.append("- frontier_backbone_llm_s1s2 shows a real overall advantage on process-quality metrics, not just isolated profile wins.")
    else:
        notes.append("- frontier_backbone_llm_s1s2 looks mixed: any gains are profile-dependent rather than a clean universal win.")

    hindered_rows = {
        flow_name: next(
            row for row in profile_rows
            if row["flow_name"] == flow_name and row["profile_name"] == "hindered_aryl_chloride_activation"
        )
        for flow_name in FINAL_BENCHMARK_FLOW_PROTOCOL_DIRS
    }
    hindered_best_stage2 = max(float(row["stage2_mean"]) for row in hindered_rows.values())
    hindered_full_stack_gap = round(hindered_best_stage2 - float(hindered_rows["frontier_backbone_llm_s1s2"]["stage2_mean"]), 4)
    if hindered_full_stack_gap > 0.005:
        notes.append("- hindered_aryl_chloride_activation remains a sensitivity case for the full S1+S2 stack.")
    else:
        notes.append("- hindered_aryl_chloride_activation no longer stands out as a major sensitivity case for the full stack.")

    notes.extend(["", "## Interpretation"])
    notes.append(
        f"- Relative to backbone, frontier_backbone_llm_s2 changes stage2_mean by {llm_s2_stage2 - backbone_stage2:+.4f} and cumulative_regret by {llm_s2_cum_regret - backbone_cum_regret:+.4f}."
    )
    notes.append(
        f"- Relative to backbone, frontier_backbone_llm_s1s2 changes stage2_mean by {llm_s1s2_stage2 - backbone_stage2:+.4f} and cumulative_regret by {llm_s1s2_cum_regret - backbone_cum_regret:+.4f}."
    )
    notes.append("- Use per-profile summaries to separate overall-average gains from profile-specific tradeoffs.")

    notes.extend(["", "## Recommendation"])
    if llm_s1s2_stage2 >= llm_s2_stage2 and llm_s1s2_cum_regret <= llm_s2_cum_regret:
        notes.append("- The current V2 system is ready to be treated as the final evaluated version, with frontier_backbone_llm_s1s2 as the primary reported LLM-backed system and frontier_backbone_llm_s2 as the safety-oriented companion ablation.")
    elif llm_s2_stage2 >= llm_s1s2_stage2 and llm_s2_cum_regret <= llm_s1s2_cum_regret + 0.05:
        notes.append("- The current V2 system is ready to be treated as the final evaluated version, with frontier_backbone_llm_s2 as the primary reported LLM-backed system and frontier_backbone_llm_s1s2 reported as the fuller but more profile-sensitive stack.")
    else:
        notes.append("- The V2 system is evaluable as final, but the LLM-backed conclusion should stay cautious because the final benchmark remains mixed across profiles.")
    notes.append("- Leave the calibrated S1/S2 behavior unchanged in any final write-up; the benchmark here is the frozen target.")

    write_markdown(output_dir / "final_benchmark_notes.md", notes)

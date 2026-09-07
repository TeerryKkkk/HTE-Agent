"""Explicit aggregation of current run summaries, preserving evaluation provenance."""
from __future__ import annotations

from collections import defaultdict
import json
import re
from pathlib import Path

from hte_agent.config import ASSISTANCE_SEMANTICS, MODES, create_flow_config
from hte_agent.shared.io_utils import ensure_dir, read_csv_rows, write_csv


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
        stage2_rows = [item for item in round_rows if str(item.get("batch_kind", "")) == "stage2_surrogate_batch"]
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
        stage2_changed_counts = [_to_int(diag.get("final_batch_changed_items", 0)) for diag in stage2_diags]
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
                "seed": row.get("seed", _protocol_seed(protocol_name)),
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
        if row.get("assistance_semantics") != ASSISTANCE_SEMANTICS:
            raise ValueError("Refusing to aggregate legacy or unknown assistance semantics with current runs")
        enriched_rows.append(enriched)
    return enriched_rows


def _collect_rows(results_root: Path, mapping: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    comparison_rows: list[dict[str, object]] = []
    for flow_name, protocol_dirs in mapping.items():
        for protocol_name, dir_name in protocol_dirs.items():
            comparison_rows.extend(_enriched_rows_for_output_dir(flow_name, protocol_name, results_root / dir_name))
    return comparison_rows


def build_comparison(results_root: Path, protocol: str) -> Path | None:
    """Aggregate available runs of one protocol, separately by evaluation status.

    Failed/unavailable LLM runs remain visible and are never pooled with successful
    LLM evaluations. This function does not write the curated benchmark snapshot.
    """
    mapping = {}
    for mode in MODES:
        config = create_flow_config(mode, protocol, results_root=results_root)
        mapping[config.flow_name] = {spec.name: output.name for spec, output in config.protocols}
    rows = _collect_rows(results_root, mapping)
    if not rows:
        return None
    output_dir = ensure_dir(results_root / f"{protocol.replace('-', '_')}_comparison")
    write_csv(output_dir / "comparison.csv", rows)
    metrics = {
        "best_observed_mean": "best_observed_yield", "oracle_regret_mean": "regret_to_oracle",
        "cumulative_regret_mean": "cumulative_regret", "stage2_mean": "stage2_batch_mean_yield",
        "round1_best_mean": "round1_batch_best_yield", "topk_true_mean": "stage2_topk_true_mean_yield",
        "post_gate_true_mean": "stage2_post_gate_batch_true_mean_yield", "gate_delta_mean": "stage2_gate_batch_quality_delta",
    }
    diagnostics = {
        "s1_reject_mean": "llm_s1_rejected_count", "s1_contradiction_mean": "llm_s1_contradiction_resolution_count",
        "s1_modified_fraction_mean": "llm_s1_modified_fraction", "s2_topk_size_mean": "llm_s2_topk_size",
        "s2_keep_mean": "llm_s2_keep_count", "s2_downweight_mean": "llm_s2_downweight_count",
        "s2_reject_mean": "llm_s2_reject_count", "s2_prefer_mean": "llm_s2_prefer_count",
        "s2_nonkeep_per_round_mean": "llm_s2_nonkeep_actions_per_round",
        "s2_changed_batch_items_mean": "llm_s2_changed_batch_items",
        "s2_mean_nonkeep_confidence": "llm_s2_mean_nonkeep_confidence",
    }
    for filename, keys, measures in (
        ("overall_summary.csv", ("flow_name", "llm_evaluation_status", "optimization_succeeded"), metrics),
        ("profile_summary.csv", ("flow_name", "profile_name", "llm_evaluation_status", "optimization_succeeded"), metrics),
        ("diagnostics_summary.csv", ("flow_name", "llm_evaluation_status", "optimization_succeeded"), diagnostics),
    ):
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
        summaries = []
        for key, items in sorted(grouped.items()):
            summary = {**dict(zip(keys, key)), "num_profile_runs": len(items), "assistance_semantics": ASSISTANCE_SEMANTICS}
            summary.update({label: round(_mean([_to_float(item.get(source, 0)) for item in items]), 4)
                            for label, source in measures.items()})
            if measures is metrics:
                summary["best_observed_std"] = round(_std([_to_float(item["best_observed_yield"]) for item in items]), 4)
                summary["stage2_std"] = round(_std([_to_float(item["stage2_batch_mean_yield"]) for item in items]), 4)
            else:
                summary["s1_extra_pool_shrinkage_mean"] = round(_mean([
                    _to_float(item["stage1_baseline_candidate_pool_after_rule_filter"]) - _to_float(item["stage1_candidate_pool_after_rule_filter"])
                    for item in items]), 4)
            for prefix in ("llm_s1", "llm_s2_summary", "llm_s2_gate"):
                summary[f"{prefix}_models"] = json.dumps(sorted({model for item in items for model in json.loads(str(item.get(f"{prefix}_models", "[]")))}))
            summaries.append(summary)
        write_csv(output_dir / filename, summaries)
    return output_dir

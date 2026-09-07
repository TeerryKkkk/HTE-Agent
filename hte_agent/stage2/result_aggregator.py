from __future__ import annotations

from collections import defaultdict

from hte_agent.shared.candidate_schema import Observation, RoundAggregate
from hte_agent.shared.io_utils import mean


def _factor_table(observations: list[Observation], key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation.parameters.get(key, ""))].append(observation)
    rows = []
    for label, items in grouped.items():
        yields = [item.yield_value for item in items]
        successes = [1.0 if item.outcome_label == "success" else 0.0 for item in items]
        rows.append(
            {
                "label": label,
                "count": len(items),
                "mean_yield": round(mean(yields), 4),
                "best_yield": round(max(yields), 4),
                "success_rate": round(mean(successes), 4),
            }
        )
    return sorted(rows, key=lambda row: (row["mean_yield"], row["count"]), reverse=True)


def aggregate_round(observations: list[Observation], round_index: int) -> RoundAggregate:
    yields = [item.yield_value for item in observations]
    selectivities = [item.selectivity for item in observations]
    success_rate = mean([1.0 if item.outcome_label == "success" else 0.0 for item in observations])
    notes: list[str] = []
    if observations:
        best = max(observations, key=lambda item: item.yield_value)
        notes.append(f"best_candidate={best.candidate_id}")
        notes.append(f"best_yield={best.yield_value:.3f}")
    return RoundAggregate(
        round_index=round_index,
        observation_count=len(observations),
        mean_yield=round(mean(yields), 4),
        best_yield=round(max(yields) if yields else 0.0, 4),
        success_rate=round(success_rate, 4),
        mean_selectivity=round(mean(selectivities), 4),
        factor_tables={
            "ligand_label": _factor_table(observations, "ligand_label"),
            "base_label": _factor_table(observations, "base_label"),
            "solvent_label": _factor_table(observations, "solvent_label"),
        },
        notes=notes,
    )

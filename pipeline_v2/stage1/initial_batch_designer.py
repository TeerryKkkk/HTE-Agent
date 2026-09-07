from __future__ import annotations

from pipeline_v2.shared.candidate_schema import Candidate
from pipeline_v2.shared.io_utils import safe_divide


def _distance(left: Candidate, right: Candidate) -> float:
    score = 0.0
    weight = 0.0
    for key, current_weight in (("ligand_label", 1.4), ("base_label", 1.3), ("solvent_label", 1.1)):
        score += current_weight * (0.0 if str(left.parameters.get(key, "")) == str(right.parameters.get(key, "")) else 1.0)
        weight += current_weight
    temperature_gap = abs(float(left.parameters["temperature_c"]) - float(right.parameters["temperature_c"])) / 40.0
    catalyst_gap = abs(float(left.parameters["catalyst_mol_pct"]) - float(right.parameters["catalyst_mol_pct"])) / 0.1
    base_equiv_gap = abs(float(left.parameters["base_equiv"]) - float(right.parameters["base_equiv"])) / 1.5
    score += min(1.0, temperature_gap) + min(1.0, catalyst_gap) + min(1.0, base_equiv_gap)
    weight += 3.0
    return safe_divide(score, weight)


def _quota_ok(candidate: Candidate, selected: list[Candidate], batch_size: int) -> bool:
    if not selected:
        return True
    max_ligand = max(8, int(batch_size * 0.35))
    max_base = max(10, int(batch_size * 0.4))
    max_solvent = max(12, int(batch_size * 0.45))
    ligand = str(candidate.parameters["ligand_label"])
    base = str(candidate.parameters["base_label"])
    solvent = str(candidate.parameters["solvent_label"])
    ligand_count = sum(1 for item in selected if str(item.parameters["ligand_label"]) == ligand)
    base_count = sum(1 for item in selected if str(item.parameters["base_label"]) == base)
    solvent_count = sum(1 for item in selected if str(item.parameters["solvent_label"]) == solvent)
    return ligand_count < max_ligand and base_count < max_base and solvent_count < max_solvent


def design_initial_batch(candidates: list[Candidate], batch_size: int) -> tuple[list[Candidate], list[dict[str, object]]]:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return [], []
    candidate_pool = feasible[: max(batch_size * 4, batch_size)]
    selected: list[Candidate] = []
    trace: list[dict[str, object]] = []

    while candidate_pool and len(selected) < batch_size:
        best_candidate: Candidate | None = None
        best_value = -1.0
        best_diversity = 0.0
        for candidate in candidate_pool:
            if not _quota_ok(candidate, selected, batch_size):
                continue
            diversity = 1.0 if not selected else min(_distance(candidate, chosen) for chosen in selected)
            selection_value = candidate.stage1_score + 0.18 * diversity
            if selection_value > best_value:
                best_candidate = candidate
                best_value = selection_value
                best_diversity = diversity
        if best_candidate is None:
            best_candidate = candidate_pool[0]
        selected.append(best_candidate)
        trace.append(
            {
                "step": len(selected),
                "candidate_id": best_candidate.candidate_id,
                "stage1_score": best_candidate.stage1_score,
                "diversity_bonus": round(0.18 * best_diversity, 4),
                "selection_value": round(best_value if best_value >= 0.0 else best_candidate.stage1_score, 4),
                "ligand_label": best_candidate.parameters["ligand_label"],
                "base_label": best_candidate.parameters["base_label"],
                "solvent_label": best_candidate.parameters["solvent_label"],
            }
        )
        candidate_pool = [candidate for candidate in candidate_pool if candidate.candidate_id != best_candidate.candidate_id]
    return selected, trace

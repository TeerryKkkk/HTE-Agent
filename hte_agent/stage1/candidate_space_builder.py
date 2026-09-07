from __future__ import annotations

import itertools
from collections import Counter

from hte_agent.shared.candidate_schema import Candidate
from hte_agent.shared.io_utils import bounded
from hte_agent.shared.task_representation import TaskRepresentation
from hte_agent.stage1.retrieval_initializer import HistoricalDataset, RetrievedSupportBundle


def _weighted_top_values(records, value_attr: str, limit: int) -> list[str]:
    counter: Counter[str] = Counter()
    for record in records:
        value = getattr(record, value_attr)
        if value and value != "unknown":
            counter[value] += max(record.retrieval_score, 0.01)
    return [label for label, _ in counter.most_common(limit)]


def _fallback_values(task: TaskRepresentation) -> dict[str, list[str]]:
    if task.profile_name == "aza_heteroaryl_chloride_guarded":
        return {
            "ligand_label": ["BrettPhos", "XPhos", "DavePhos", "JohnPhos", "GPhos", "AlPhos"],
            "base_label": ["DBU", "K3PO4", "Cs2CO3", "LiHMDS", "NaOTMS"],
            "solvent_label": ["Dioxane", "THF", "DMAc", "Toluene"],
        }
    if task.profile_name == "hindered_aryl_chloride_activation":
        return {
            "ligand_label": ["tBuBrettPhos", "AlPhos", "GPhos", "BrettPhos", "XPhos", "SPhos"],
            "base_label": ["Cs2CO3", "NaOTMS", "K3PO4", "DBU", "LiHMDS"],
            "solvent_label": ["Dioxane", "Toluene", "THF", "DMAc"],
        }
    return {
        "ligand_label": ["XPhos", "SPhos", "BrettPhos", "GPhos", "JohnPhos"],
        "base_label": ["Cs2CO3", "K3PO4", "DBU", "NaOTMS", "LiHMDS"],
        "solvent_label": ["Dioxane", "Toluene", "THF", "MeCN"],
    }


def _temperature_grid(task: TaskRepresentation) -> list[float]:
    if task.profile_name == "aza_heteroaryl_chloride_guarded":
        return [65.0, 78.0, 90.0]
    if task.profile_name == "hindered_aryl_chloride_activation":
        return [85.0, 96.0, 105.0]
    return [75.0, 88.0, 98.0]


def build_candidate_universe(
    task: TaskRepresentation,
    support: RetrievedSupportBundle,
    dataset: HistoricalDataset,
    candidate_pool_cap: int,
) -> list[Candidate]:
    fallback = _fallback_values(task)
    ligands = _weighted_top_values((*support.precedents, *support.success_cases), "ligand_label", 6) or fallback["ligand_label"]
    bases = _weighted_top_values((*support.precedents, *support.success_cases, *support.failure_cases), "base_label", 5) or fallback["base_label"]
    solvents = _weighted_top_values((*support.precedents, *support.success_cases), "solvent_label", 4) or fallback["solvent_label"]
    for axis, values in (("ligand_label", ligands), ("base_label", bases), ("solvent_label", solvents)):
        for item in fallback[axis]:
            if item not in values:
                values.append(item)
    temperature_grid = _temperature_grid(task)
    catalyst_grid = [0.10, 0.15, 0.20]
    base_equiv_grid = [1.5, 2.0, 3.0]

    candidates: list[Candidate] = []
    ordered_space = itertools.product(ligands[:6], bases[:5], solvents[:4], temperature_grid, catalyst_grid, base_equiv_grid)
    for index, (ligand, base, solvent, temperature, catalyst_mol_pct, base_equiv) in enumerate(ordered_space, start=1):
        if index > candidate_pool_cap:
            break
        candidates.append(
            Candidate(
                candidate_id=f"{task.profile_name}-cand-{index:04d}",
                parameters={
                    "ligand_label": ligand,
                    "ligand_family": dataset.family_lookup["ligand_label"].get(ligand, "unknown"),
                    "base_label": base,
                    "base_family": dataset.family_lookup["base_label"].get(base, "unknown"),
                    "solvent_label": solvent,
                    "solvent_family": dataset.family_lookup["solvent_label"].get(solvent, "unknown"),
                    "temperature_c": bounded(float(temperature), 50.0, 110.0),
                    "catalyst_mol_pct": float(catalyst_mol_pct),
                    "base_equiv": float(base_equiv),
                },
            )
        )
    return candidates

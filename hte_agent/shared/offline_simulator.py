from __future__ import annotations

from hashlib import md5

from hte_agent.shared.candidate_schema import Candidate, Observation
from hte_agent.shared.task_representation import TaskRepresentation
from hte_agent.shared.io_utils import bounded


_VISIBLE_LIGAND_EFFECT = {
    "BrettPhos": 0.10,
    "tBuBrettPhos": 0.10,
    "XPhos": 0.09,
    "SPhos": 0.07,
    "GPhos": 0.07,
    "AlPhos": 0.07,
    "DavePhos": 0.06,
    "JohnPhos": 0.05,
    "XantPhos": 0.02,
    "BINAP": 0.01,
}
_VISIBLE_BASE_EFFECT = {
    "Cs2CO3": 0.08,
    "K3PO4": 0.07,
    "DBU": 0.06,
    "DIPEA": 0.04,
    "NaOTMS": 0.03,
    "LiHMDS": -0.01,
    "NaOtBu": -0.04,
    "KOtBu": -0.05,
}
_VISIBLE_SOLVENT_EFFECT = {
    "Dioxane": 0.08,
    "THF": 0.07,
    "Toluene": 0.05,
    "DMAc": 0.04,
    "MeCN": 0.04,
    "DMF": 0.03,
    "MeOH": -0.04,
    "Water": -0.06,
}


def _profile_targets(task: TaskRepresentation) -> dict[str, object]:
    if task.profile_name == "aza_heteroaryl_chloride_guarded":
        return {
            "ligands": {"BrettPhos", "XPhos"},
            "bases": {"DBU", "K3PO4"},
            "solvents": {"Dioxane", "THF"},
            "temperature_center": 78.0,
            "hidden_synergy": ("BrettPhos", "DBU", "Dioxane"),
            "hidden_failure": ("XPhos", "K3PO4", "DMAc"),
        }
    if task.profile_name == "hindered_aryl_chloride_activation":
        return {
            "ligands": {"tBuBrettPhos", "AlPhos", "GPhos"},
            "bases": {"Cs2CO3", "NaOTMS"},
            "solvents": {"Dioxane", "Toluene"},
            "temperature_center": 96.0,
            "hidden_synergy": ("tBuBrettPhos", "Cs2CO3", "Dioxane"),
            "hidden_failure": ("AlPhos", "DBU", "THF"),
        }
    return {
        "ligands": {"XPhos", "SPhos"},
        "bases": {"Cs2CO3", "K3PO4"},
        "solvents": {"Dioxane", "Toluene"},
        "temperature_center": 88.0,
        "hidden_synergy": ("SPhos", "Cs2CO3", "Toluene"),
        "hidden_failure": ("XPhos", "DBU", "MeCN"),
    }


def _noise(candidate_id: str, profile_name: str, seed: int) -> float:
    digest = md5(f"{seed}:{profile_name}:{candidate_id}".encode("utf-8")).hexdigest()
    value = int(digest[:6], 16) / 0xFFFFFF
    return round((value - 0.5) * 0.06, 4)


def simulate_observation(candidate: Candidate, task: TaskRepresentation, seed: int, round_index: int) -> Observation:
    params = candidate.parameters
    ligand = str(params.get("ligand_label", ""))
    base = str(params.get("base_label", ""))
    solvent = str(params.get("solvent_label", ""))
    temperature = float(params.get("temperature_c", 80.0))
    catalyst_loading = float(params.get("catalyst_mol_pct", 0.15))
    base_equiv = float(params.get("base_equiv", 2.0))
    targets = _profile_targets(task)

    visible_signal = 0.28
    visible_signal += _VISIBLE_LIGAND_EFFECT.get(ligand, 0.02)
    visible_signal += _VISIBLE_BASE_EFFECT.get(base, 0.01)
    visible_signal += _VISIBLE_SOLVENT_EFFECT.get(solvent, 0.01)
    visible_signal += max(-0.05, 0.06 - abs(0.15 - catalyst_loading) / 1.8)
    visible_signal += max(-0.05, 0.04 - abs(2.0 - base_equiv) / 3.0)

    hidden_bonus = 0.0
    if ligand in targets["ligands"]:
        hidden_bonus += 0.06
    if base in targets["bases"]:
        hidden_bonus += 0.05
    if solvent in targets["solvents"]:
        hidden_bonus += 0.04
    temperature_center = float(targets["temperature_center"])
    hidden_bonus += max(-0.08, 0.09 - abs(temperature_center - temperature) / 120.0)
    if (ligand, base, solvent) == tuple(targets["hidden_synergy"]):
        hidden_bonus += 0.12
    if (ligand, base, solvent) == tuple(targets["hidden_failure"]):
        hidden_bonus -= 0.15

    if "base_sensitive" in task.substrate_features and base in {"NaOtBu", "KOtBu", "LiHMDS"}:
        hidden_bonus -= 0.14
    if "heteroaryl" in task.substrate_features and solvent in {"DMAc", "DMF"} and base in {"K3PO4", "DBU"}:
        hidden_bonus -= 0.07
    if "high_activation_barrier" in task.substrate_features and ligand not in {"tBuBrettPhos", "AlPhos", "GPhos"}:
        hidden_bonus -= 0.08

    noise = _noise(candidate.candidate_id, task.profile_name, seed)
    yield_value = round(bounded(visible_signal + hidden_bonus + noise, 0.03, 0.97), 3)

    selectivity = 0.88
    selectivity -= abs(temperature_center - temperature) / 180.0
    if base in {"NaOtBu", "KOtBu", "LiHMDS"}:
        selectivity -= 0.08
    if solvent in {"DMAc", "DMF"} and "heteroaryl" in task.substrate_features:
        selectivity -= 0.06
    selectivity += noise / 2.0
    selectivity = round(bounded(selectivity, 0.42, 0.96), 3)

    outcome_label = "success" if yield_value >= 0.72 else "mixed" if yield_value >= 0.45 else "failure"
    return Observation(
        candidate_id=candidate.candidate_id,
        round_index=round_index,
        parameters=dict(candidate.parameters),
        yield_value=yield_value,
        selectivity=selectivity,
        outcome_label=outcome_label,
        raw_measurements={
            "conversion": round(min(yield_value + 0.07, 0.99), 3),
            "impurity_index": round(1.0 - selectivity, 3),
            "hidden_bonus_proxy": round(hidden_bonus, 3),
            "measurement_noise": round(noise, 3),
        },
    )


def compute_oracle_summary(candidates: list[Candidate], task: TaskRepresentation, seed: int) -> dict[str, object]:
    ranked = sorted(
        (
            (simulate_observation(candidate, task, seed, round_index=0).yield_value, candidate)
            for candidate in candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_yield, best_candidate = ranked[0]
    top5 = ranked[:5]
    return {
        "oracle_best_yield": round(best_yield, 4),
        "oracle_best_candidate_id": best_candidate.candidate_id,
        "oracle_top5_floor": round(top5[-1][0], 4),
        "oracle_top5_candidate_ids": [candidate.candidate_id for _, candidate in top5],
        "candidate_count": len(candidates),
    }

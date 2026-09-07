from __future__ import annotations

from hte_agent.shared.candidate_schema import Candidate, LLMS1Critique, model_response_succeeded
from hte_agent.shared.task_representation import TaskRepresentation


def _combination_matches(candidate: Candidate, combination: dict[str, str]) -> bool:
    return all(str(candidate.parameters.get(key, "")) == str(value) for key, value in combination.items())


def apply_rule_filter(
    task: TaskRepresentation,
    candidates: list[Candidate],
    llm_critique: LLMS1Critique | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    if llm_critique and not model_response_succeeded(llm_critique):
        llm_critique = None
    feasible: list[Candidate] = []
    rejected: list[Candidate] = []
    llm_reject_values = llm_critique.reject_values if llm_critique else {}
    llm_reject_combinations = llm_critique.reject_combinations if llm_critique else []
    llm_reject_value_reason_labels = llm_critique.reject_value_reason_labels if llm_critique else {}
    llm_reject_combination_reason_labels = llm_critique.reject_combination_reason_labels if llm_critique else []

    for candidate in candidates:
        params = candidate.parameters
        base = str(params.get("base_label", ""))
        solvent = str(params.get("solvent_label", ""))
        temperature = float(params.get("temperature_c", 80.0))
        reasons: list[str] = []

        if solvent in {"Water", "MeOH"}:
            reasons.append("nonviable_protic_or_aqueous_solvent")
        if "base_sensitive" in task.substrate_features and base in {"NaOtBu", "KOtBu"}:
            reasons.append("strong_base_conflict_with_base_sensitive_profile")
        if task.risk_profile == "guarded" and temperature > 100.0:
            reasons.append("temperature_above_guarded_limit")
        if task.profile_name == "aza_heteroaryl_chloride_guarded" and solvent in {"DMAc", "DMF"} and base in {"DBU", "K3PO4"}:
            reasons.append("heteroaryl_poisoning_risk_window")
        if task.profile_name == "hindered_aryl_chloride_activation" and temperature < 80.0 and str(params.get("ligand_label", "")) not in {"tBuBrettPhos", "AlPhos", "GPhos"}:
            reasons.append("underactivated_low_temperature_regime")
        for key, rejected_values in llm_reject_values.items():
            if str(params.get(key, "")) in {str(value) for value in rejected_values}:
                reason_label = str(llm_reject_value_reason_labels.get(key, {}).get(str(params.get(key, "")), "conflict"))
                reasons.append(f"llm_reject_value:{reason_label}:{key}={params.get(key, '')}")
        for index, combination in enumerate(llm_reject_combinations):
            if _combination_matches(candidate, combination):
                reason_label = str(llm_reject_combination_reason_labels[index] if index < len(llm_reject_combination_reason_labels) else "conflict")
                reasons.append(f"llm_reject_combination:{reason_label}")
                break

        if reasons:
            candidate.feasible = False
            candidate.rejection_reasons = reasons
            rejected.append(candidate)
        else:
            candidate.feasible = True
            feasible.append(candidate)
    return feasible, rejected

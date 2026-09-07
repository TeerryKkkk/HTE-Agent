from __future__ import annotations

from collections import defaultdict

from hte_agent.shared.candidate_schema import Candidate, LLMS1Critique, model_response_succeeded, PrecedentRecord
from hte_agent.shared.io_utils import bounded, mean, safe_divide
from hte_agent.shared.task_representation import TaskRepresentation
from hte_agent.stage1.retrieval_initializer import RetrievedSupportBundle


def _continuous_similarity(left: float, right: float, span: float) -> float:
    return max(0.0, 1.0 - abs(left - right) / max(span, 1.0))


def _record_similarity(candidate: Candidate, record: PrecedentRecord) -> float:
    params = candidate.parameters
    score = 0.0
    weight = 0.0
    for key, current_weight in (("ligand_label", 1.5), ("base_label", 1.4), ("solvent_label", 1.2)):
        score += current_weight * (1.0 if str(params.get(key, "")) == str(getattr(record, key)) else 0.0)
        weight += current_weight
    score += 1.0 * _continuous_similarity(float(params["temperature_c"]), record.temperature_c, 35.0)
    weight += 1.0
    return safe_divide(score, weight)


def _family_support(candidate: Candidate, records: tuple[PrecedentRecord, ...]) -> float:
    if not records:
        return 0.0
    ligand_family = str(candidate.parameters.get("ligand_family", "unknown"))
    base_family = str(candidate.parameters.get("base_family", "unknown"))
    solvent_family = str(candidate.parameters.get("solvent_family", "unknown"))
    signals: list[float] = []
    for record in records:
        score = 0.0
        if record.ligand_family == ligand_family:
            score += 0.45
        if record.base_family == base_family:
            score += 0.35
        if record.solvent_family == solvent_family:
            score += 0.20
        signals.append(score * max(record.outcome_yield, 0.15))
    return mean(signals)


def _continuous_prior(task: TaskRepresentation, candidate: Candidate) -> float:
    temperature = float(candidate.parameters["temperature_c"])
    catalyst_mol_pct = float(candidate.parameters["catalyst_mol_pct"])
    base_equiv = float(candidate.parameters["base_equiv"])
    target_temperature = 78.0 if task.profile_name == "aza_heteroaryl_chloride_guarded" else 96.0 if task.profile_name == "hindered_aryl_chloride_activation" else 88.0
    score = _continuous_similarity(temperature, target_temperature, 22.0)
    score += _continuous_similarity(catalyst_mol_pct, 0.15, 0.10)
    score += _continuous_similarity(base_equiv, 2.0, 1.5)
    return safe_divide(score, 3.0)


def _bonus_reason_label(llm_critique: LLMS1Critique, axis: str, value: str) -> str:
    return str(llm_critique.bonus_reason_labels.get(axis, {}).get(value, "positive_evidence"))


def _penalty_reason_label(llm_critique: LLMS1Critique, axis: str, value: str) -> str:
    if value in llm_critique.penalty_reason_labels.get(axis, {}):
        return str(llm_critique.penalty_reason_labels[axis][value])
    if axis == "ligand_label":
        return "weak_transfer"
    if axis in {"base_label", "solvent_label"}:
        return "dubious_compatibility"
    return "low_support"


def score_candidates(
    task: TaskRepresentation,
    candidates: list[Candidate],
    support: RetrievedSupportBundle,
    llm_critique: LLMS1Critique | None = None,
) -> list[Candidate]:
    success_records = tuple((*support.precedents, *support.success_cases))
    failure_records = support.failure_cases
    scored_candidates: list[Candidate] = []
    for candidate in candidates:
        support_weights = [
            _record_similarity(candidate, record) * record.outcome_yield * max(record.retrieval_score, 0.05)
            for record in success_records
        ]
        similarity_weighted_support = mean(support_weights)
        success_match = mean([_record_similarity(candidate, record) for record in support.success_cases])
        failure_match = mean([_record_similarity(candidate, record) for record in failure_records])
        success_failure_contrast = bounded(0.5 + 0.8 * (success_match - failure_match), 0.0, 1.0)
        family_support = bounded(_family_support(candidate, success_records), 0.0, 1.0)
        continuous_prior = _continuous_prior(task, candidate)
        base_score = 0.40 * similarity_weighted_support + 0.28 * success_failure_contrast + 0.18 * family_support + 0.14 * continuous_prior

        llm_delta = 0.0
        llm_flags: list[str] = []
        if llm_critique and model_response_succeeded(llm_critique):
            for key, adjustments in llm_critique.bonuses.items():
                value = str(candidate.parameters.get(key, ""))
                if value in adjustments:
                    llm_delta += float(adjustments[value])
                    llm_flags.append(f"bonus:{_bonus_reason_label(llm_critique, key, value)}:{key}={value}")
            for key, adjustments in llm_critique.penalties.items():
                value = str(candidate.parameters.get(key, ""))
                if value in adjustments:
                    llm_delta -= float(adjustments[value])
                    llm_flags.append(f"penalty:{_penalty_reason_label(llm_critique, key, value)}:{key}={value}")
            llm_delta = bounded(
                llm_delta,
                -float(llm_critique.diagnostics.get("max_candidate_penalty_magnitude", 0.06)),
                float(llm_critique.diagnostics.get("max_candidate_bonus_magnitude", 0.03)),
            )

        candidate.stage1_components = {
            "similarity_weighted_support": round(similarity_weighted_support, 4),
            "success_failure_contrast": round(success_failure_contrast, 4),
            "family_support": round(family_support, 4),
            "continuous_prior": round(continuous_prior, 4),
            "llm_delta": round(llm_delta, 4),
        }
        candidate.stage1_score = round(base_score + llm_delta, 4)
        candidate.llm_stage1_delta = round(llm_delta, 4)
        candidate.llm_stage1_flags = llm_flags
        candidate.source_support = [
            f"success_match={success_match:.3f}",
            f"failure_match={failure_match:.3f}",
            f"family_support={family_support:.3f}",
        ]
        scored_candidates.append(candidate)

    scored_candidates.sort(key=lambda item: item.stage1_score, reverse=True)
    for rank, candidate in enumerate(scored_candidates, start=1):
        candidate.stage1_rank = rank
    return scored_candidates


def summarize_transfer_axes(candidates: list[Candidate], limit: int = 8) -> dict[str, list[dict[str, object]]]:
    counters: dict[str, defaultdict[str, float]] = {
        "ligand_label": defaultdict(float),
        "base_label": defaultdict(float),
        "solvent_label": defaultdict(float),
    }
    for candidate in candidates:
        for axis in counters:
            counters[axis][str(candidate.parameters.get(axis, ""))] += candidate.stage1_score
    return {
        axis: [{"label": label, "score": round(score, 4)} for label, score in sorted(counter.items(), key=lambda item: item[1], reverse=True)[:limit]]
        for axis, counter in counters.items()
    }

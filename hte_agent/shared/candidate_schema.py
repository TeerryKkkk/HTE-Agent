from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ParameterValue = str | int | float


@dataclass
class PrecedentRecord:
    record_id: str
    source_kind: str
    reaction_smiles: str
    substrate_archetype: str
    product_archetype: str
    substrate_features: list[str]
    product_features: list[str]
    ligand_label: str
    ligand_family: str
    base_label: str
    base_family: str
    solvent_label: str
    solvent_family: str
    temperature_c: float
    time_h: float
    outcome_label: str
    outcome_yield: float
    confidence: float
    provenance: str
    raw_record: dict[str, str] = field(default_factory=dict)
    retrieval_score: float = 0.0
    retrieval_notes: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    candidate_id: str
    parameters: dict[str, ParameterValue]
    stage1_components: dict[str, float] = field(default_factory=dict)
    stage1_score: float = 0.0
    stage1_rank: int = 0
    feasible: bool = True
    rejection_reasons: list[str] = field(default_factory=list)
    source_support: list[str] = field(default_factory=list)
    llm_stage1_delta: float = 0.0
    llm_stage1_flags: list[str] = field(default_factory=list)
    trust_region_bonus: float = 0.0
    round_selected: int | None = None


@dataclass
class Observation:
    candidate_id: str
    round_index: int
    parameters: dict[str, ParameterValue]
    yield_value: float
    selectivity: float
    outcome_label: str
    raw_measurements: dict[str, float] = field(default_factory=dict)


@dataclass
class Stage2Score:
    candidate_id: str
    predicted_mean: float
    predicted_std: float
    acquisition_score: float
    prior_score: float
    trust_region_bonus: float
    feasibility_penalty: float
    llm_delta: float = 0.0
    llm_decision: str = "not_applicable"
    llm_reason: str = ""
    final_score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class LLMS1Critique:
    invoked: bool
    status: str
    model: str
    summary: str
    bonuses: dict[str, dict[str, float]] = field(default_factory=dict)
    penalties: dict[str, dict[str, float]] = field(default_factory=dict)
    reject_values: dict[str, list[str]] = field(default_factory=dict)
    reject_combinations: list[dict[str, str]] = field(default_factory=list)
    bonus_reason_labels: dict[str, dict[str, str]] = field(default_factory=dict)
    penalty_reason_labels: dict[str, dict[str, str]] = field(default_factory=dict)
    reject_value_reason_labels: dict[str, dict[str, str]] = field(default_factory=dict)
    reject_combination_reason_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMS2Summary:
    invoked: bool
    status: str
    model: str
    summary: str
    promising_patterns: list[str] = field(default_factory=list)
    risky_patterns: list[str] = field(default_factory=list)
    exploration_axes: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMS2Gate:
    invoked: bool
    status: str
    model: str
    summary: str
    candidate_actions: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustRegion:
    preferred_values: dict[str, list[str]]
    temperature_window: tuple[float, float]
    catalyst_window: tuple[float, float]
    base_equiv_window: tuple[float, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class RoundAggregate:
    round_index: int
    observation_count: int
    mean_yield: float
    best_yield: float
    success_rate: float
    mean_selectivity: float
    factor_tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def model_response_succeeded(result: LLMS1Critique | LLMS2Summary | LLMS2Gate) -> bool:
    """Only a successfully invoked, parsed model response can authorize LLM actions."""
    return result.invoked and result.status in {"success", "success_calibrated"}

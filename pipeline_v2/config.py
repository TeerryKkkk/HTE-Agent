from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT = DATA_ROOT / "Datas"
RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class SharedRunConstants:
    """Backbone settings shared by the two V2 flows."""

    reaction_family: str = "buchwald-hartwig-amination"
    objective: str = "maximize isolated yield while respecting guarded screening constraints"
    precedent_pool_size: int = 48
    support_case_pool_size: int = 48
    candidate_pool_cap: int = 1200
    initial_batch_size: int = 96
    stage2_batch_size: int = 12
    stage2_rounds: int = 1
    topk_rerank_pool_size: int = 18
    trust_region_pool_cap: int = 240
    random_seed: int = 7
    validation_seed: int = 11
    benchmark_seeds: tuple[int, ...] = (17, 23)
    final_benchmark_seeds: tuple[int, ...] = (13, 17, 19, 23, 29, 31)
    medium_benchmark_stage2_rounds: int = 2
    final_benchmark_stage2_rounds: int = 2
    acquisition_beta: float = 0.22
    diversity_weight: float = 0.18
    trust_region_bonus: float = 0.06
    prior_blend: float = 0.38
    llm_score_cap: float = 0.18
    s1_max_bonus_magnitude: float = 0.03
    s1_max_penalty_magnitude: float = 0.05
    s1_contradiction_penalty_magnitude: float = 0.02
    s1_min_effect_to_keep: float = 0.015
    s1_max_modified_fraction: float = 0.32
    s1_max_reject_fraction: float = 0.005
    s1_max_single_entry_fraction: float = 0.22
    s1_max_bonus_entries: int = 2
    s1_max_penalty_entries: int = 3
    s1_max_reject_value_entries: int = 0
    s1_max_reject_combination_entries: int = 1
    s2_prefer_delta: float = 0.02
    s2_downweight_delta: float = -0.04
    s2_reject_delta: float = -0.08
    s2_min_abs_delta: float = 0.02
    s2_min_confidence_for_nonkeep: float = 0.78
    s2_min_confidence_for_reject: float = 0.92
    s2_hindered_min_confidence_for_nonkeep: float = 0.84
    s2_hindered_min_confidence_for_reject: float = 0.97
    s2_min_predicted_std_for_exploration: float = 0.16
    s2_max_nonkeep_actions: int = 3
    s2_max_downweight_actions: int = 2
    s2_max_prefer_actions: int = 1
    s2_max_reject_actions: int = 1
    s2_max_final_batch_changes: int = 2
    s2_hindered_max_final_batch_changes: int = 1
    s2_boundary_band: int = 3
    s2_hindered_boundary_band: int = 2
    supported_profiles: tuple[str, ...] = (
        "aza_heteroaryl_chloride_guarded",
        "hindered_aryl_chloride_activation",
        "bromothiophene_screen",
    )


@dataclass(frozen=True)
class ProtocolConfig:
    """One intentionally small run protocol."""

    name: str
    profiles: tuple[str, ...]
    seed: int
    initial_batch_size: int
    stage2_batch_size: int
    stage2_rounds: int


@dataclass(frozen=True)
class LLMSettings:
    """OpenRouter settings used by the light-touch JSON-only LLM helpers."""

    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout_seconds: float = 60.0
    model_fallbacks: tuple[str, ...] = (
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.5-haiku",
    )
    max_precedents_in_prompt: int = 10
    max_candidates_in_prompt: int = 12


@dataclass(frozen=True)
class FlowConfig:
    """Complete run configuration for one flow."""

    flow_name: str
    llm_stage1_enabled: bool
    llm_stage2_enabled: bool
    smoke_output_dir: Path
    validation_output_dir: Path
    include_default_protocols: bool = True
    extra_protocols: tuple[tuple[ProtocolConfig, Path], ...] = ()
    constants: SharedRunConstants = field(default_factory=SharedRunConstants)
    llm: LLMSettings = field(default_factory=LLMSettings)

    @property
    def protocol_specs(self) -> tuple[ProtocolConfig, ...]:
        constants = self.constants
        default_protocols: tuple[ProtocolConfig, ...] = ()
        if self.include_default_protocols:
            default_protocols = (
                ProtocolConfig(
                    name="smoke",
                    profiles=("aza_heteroaryl_chloride_guarded",),
                    seed=constants.random_seed,
                    initial_batch_size=constants.initial_batch_size,
                    stage2_batch_size=constants.stage2_batch_size,
                    stage2_rounds=constants.stage2_rounds,
                ),
                ProtocolConfig(
                    name="small_validation",
                    profiles=(
                        "aza_heteroaryl_chloride_guarded",
                        "hindered_aryl_chloride_activation",
                    ),
                    seed=constants.validation_seed,
                    initial_batch_size=constants.initial_batch_size,
                    stage2_batch_size=constants.stage2_batch_size,
                    stage2_rounds=constants.stage2_rounds,
                ),
            )
        return default_protocols + tuple(protocol for protocol, _ in self.extra_protocols)

    def output_dir_for_protocol(self, protocol_name: str) -> Path:
        if protocol_name == "smoke":
            return self.smoke_output_dir
        if protocol_name == "small_validation":
            return self.validation_output_dir
        for protocol, output_dir in self.extra_protocols:
            if protocol.name == protocol_name:
                return output_dir
        raise KeyError(f"No output directory configured for protocol '{protocol_name}'")


def backbone_flow_config() -> FlowConfig:
    return FlowConfig(
        flow_name="frontier_backbone",
        llm_stage1_enabled=False,
        llm_stage2_enabled=False,
        smoke_output_dir=RESULTS_ROOT / "frontier_backbone_smoke",
        validation_output_dir=RESULTS_ROOT / "frontier_backbone_small_validation",
    )


def llm_flow_config() -> FlowConfig:
    return FlowConfig(
        flow_name="frontier_backbone_llm_s1s2",
        llm_stage1_enabled=True,
        llm_stage2_enabled=True,
        smoke_output_dir=RESULTS_ROOT / "frontier_backbone_llm_s1s2_smoke",
        validation_output_dir=RESULTS_ROOT / "frontier_backbone_llm_s1s2_small_validation",
    )


def llm_s1_flow_config() -> FlowConfig:
    return FlowConfig(
        flow_name="frontier_backbone_llm_s1",
        llm_stage1_enabled=True,
        llm_stage2_enabled=False,
        smoke_output_dir=RESULTS_ROOT / "frontier_backbone_llm_s1_smoke",
        validation_output_dir=RESULTS_ROOT / "frontier_backbone_llm_s1_small_validation",
    )


def llm_s2_flow_config() -> FlowConfig:
    return FlowConfig(
        flow_name="frontier_backbone_llm_s2",
        llm_stage1_enabled=False,
        llm_stage2_enabled=True,
        smoke_output_dir=RESULTS_ROOT / "frontier_backbone_llm_s2_smoke",
        validation_output_dir=RESULTS_ROOT / "frontier_backbone_llm_s2_small_validation",
    )


def _medium_benchmark_protocols(flow_name: str, constants: SharedRunConstants) -> tuple[tuple[ProtocolConfig, Path], ...]:
    return tuple(
        (
            ProtocolConfig(
                name=f"medium_benchmark_seed{seed}",
                profiles=constants.supported_profiles,
                seed=seed,
                initial_batch_size=constants.initial_batch_size,
                stage2_batch_size=constants.stage2_batch_size,
                stage2_rounds=constants.medium_benchmark_stage2_rounds,
            ),
            RESULTS_ROOT / f"{flow_name}_medium_benchmark_seed{seed}",
        )
        for seed in constants.benchmark_seeds
    )


def _final_benchmark_protocols(flow_name: str, constants: SharedRunConstants) -> tuple[tuple[ProtocolConfig, Path], ...]:
    return tuple(
        (
            ProtocolConfig(
                name=f"final_benchmark_seed{seed}",
                profiles=constants.supported_profiles,
                seed=seed,
                initial_batch_size=constants.initial_batch_size,
                stage2_batch_size=constants.stage2_batch_size,
                stage2_rounds=constants.final_benchmark_stage2_rounds,
            ),
            RESULTS_ROOT / f"{flow_name}_final_benchmark_seed{seed}",
        )
        for seed in constants.final_benchmark_seeds
    )


def backbone_medium_benchmark_flow_config() -> FlowConfig:
    base = backbone_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_medium_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )


def llm_s1_medium_benchmark_flow_config() -> FlowConfig:
    base = llm_s1_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_medium_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )


def llm_s2_medium_benchmark_flow_config() -> FlowConfig:
    base = llm_s2_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_medium_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )


def llm_s1s2_medium_benchmark_flow_config() -> FlowConfig:
    base = llm_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_medium_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )


def backbone_final_benchmark_flow_config() -> FlowConfig:
    base = backbone_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_final_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )


def llm_s1_final_benchmark_flow_config() -> FlowConfig:
    base = llm_s1_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_final_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )


def llm_s2_final_benchmark_flow_config() -> FlowConfig:
    base = llm_s2_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_final_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )


def llm_s1s2_final_benchmark_flow_config() -> FlowConfig:
    base = llm_flow_config()
    return FlowConfig(
        flow_name=base.flow_name,
        llm_stage1_enabled=base.llm_stage1_enabled,
        llm_stage2_enabled=base.llm_stage2_enabled,
        smoke_output_dir=base.smoke_output_dir,
        validation_output_dir=base.validation_output_dir,
        include_default_protocols=False,
        extra_protocols=_final_benchmark_protocols(base.flow_name, base.constants),
        constants=base.constants,
        llm=base.llm,
    )

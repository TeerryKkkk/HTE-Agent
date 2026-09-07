from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT = DATA_ROOT / "buchwald_hartwig"
RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class SharedRunConstants:
    """Shared optimizer and correction limits for all assistance modes."""

    reaction_family: str = "buchwald-hartwig-amination"
    objective: str = "maximize simulated yield while respecting guarded screening constraints"
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
    quick_benchmark_seeds: tuple[int, ...] = (17, 23)
    benchmark_seeds: tuple[int, ...] = (13, 17, 19, 23, 29, 31)
    quick_benchmark_stage2_rounds: int = 2
    benchmark_stage2_rounds: int = 2
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


MODES = ("baseline", "llm-s1", "llm-s2", "llm-s1s2")
PROTOCOLS = ("smoke", "validation", "quick-benchmark", "benchmark")
ASSISTANCE_SEMANTICS = "model_actions_only_noop_on_failure"


@dataclass(frozen=True)
class FlowConfig:
    """A mode and its explicit protocol/output pairs."""

    flow_name: str
    llm_stage1_enabled: bool
    llm_stage2_enabled: bool
    protocols: tuple[tuple[ProtocolConfig, Path], ...]
    constants: SharedRunConstants = field(default_factory=SharedRunConstants)
    llm: LLMSettings = field(default_factory=LLMSettings)

    @property
    def protocol_specs(self) -> tuple[ProtocolConfig, ...]:
        return tuple(protocol for protocol, _ in self.protocols)

    def output_dir_for_protocol(self, protocol_name: str) -> Path:
        for protocol, output_dir in self.protocols:
            if protocol.name == protocol_name:
                return output_dir
        raise KeyError(f"No output directory configured for protocol '{protocol_name}'")


def create_flow_config(
    mode: str = "baseline",
    protocol: str = "smoke",
    *,
    results_root: Path = RESULTS_ROOT,
    constants: SharedRunConstants | None = None,
    llm: LLMSettings | None = None,
) -> FlowConfig:
    """Construct smoke, validation, two-seed or six-seed simulator protocols."""
    if mode not in MODES or protocol not in PROTOCOLS:
        raise ValueError(f"Unknown mode/protocol: {mode}/{protocol}")
    constants = constants or SharedRunConstants()
    if protocol == "smoke":
        seeds = (constants.random_seed,)
        profiles = constants.supported_profiles[:1]
        rounds = constants.stage2_rounds
    elif protocol == "validation":
        seeds = (constants.validation_seed,)
        profiles = constants.supported_profiles[:2]
        rounds = constants.stage2_rounds
    else:
        seeds = constants.quick_benchmark_seeds if protocol == "quick-benchmark" else constants.benchmark_seeds
        profiles = constants.supported_profiles
        rounds = constants.quick_benchmark_stage2_rounds if protocol == "quick-benchmark" else constants.benchmark_stage2_rounds
    flow_name = mode.replace("-", "_")
    specs = []
    for seed in seeds:
        name = f"{protocol.replace('-', '_')}_seed{seed}" if "benchmark" in protocol else protocol
        spec = ProtocolConfig(name, profiles, seed, constants.initial_batch_size, constants.stage2_batch_size, rounds)
        specs.append((spec, Path(results_root) / f"{flow_name}_{name}"))
    return FlowConfig(flow_name, mode in ("llm-s1", "llm-s1s2"), mode in ("llm-s2", "llm-s1s2"),
                      tuple(specs), constants, llm or LLMSettings())

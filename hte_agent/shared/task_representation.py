from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskRepresentation:
    """High-level task summary used across Stage 1 and Stage 2."""

    profile_name: str
    reaction_family: str
    objective: str
    substrate: str
    product: str
    substrate_features: tuple[str, ...]
    product_features: tuple[str, ...]
    known_incompatibilities: tuple[str, ...]
    constraints: tuple[str, ...]
    desired_axes: tuple[str, ...] = field(default_factory=tuple)
    risk_profile: str = "balanced"
    metadata: dict[str, str] = field(default_factory=dict)


_PROFILE_LIBRARY: dict[str, dict[str, object]] = {
    "aza_heteroaryl_chloride_guarded": {
        "substrate": "aza-heteroaryl chloride electrophile",
        "product": "aza-heteroaryl arylamine",
        "substrate_features": (
            "aryl_chloride",
            "heteroaryl",
            "azaheteroaryl",
            "base_sensitive",
        ),
        "product_features": ("arylamine", "heteroaryl_product"),
        "known_incompatibilities": (
            "strong alkoxide bases at elevated temperature",
            "polar amide solvents combined with heteroaryl poisoning risk",
        ),
        "constraints": (
            "avoid extreme temperatures",
            "prefer operationally simple screening points",
            "guard against clearly failure-dominated strong-base regimes",
        ),
        "desired_axes": ("ligand", "base", "solvent", "temperature"),
        "risk_profile": "guarded",
    },
    "hindered_aryl_chloride_activation": {
        "substrate": "sterically hindered aryl chloride electrophile",
        "product": "hindered arylamine",
        "substrate_features": (
            "aryl_chloride",
            "sterically_hindered",
            "high_activation_barrier",
        ),
        "product_features": ("arylamine",),
        "known_incompatibilities": (
            "underactivated ligand sets at low temperature",
            "very weak bases for demanding oxidative addition",
        ),
        "constraints": (
            "favor realistic activation windows",
            "keep search broad enough to avoid single-family collapse",
        ),
        "desired_axes": ("ligand", "base", "solvent", "temperature"),
        "risk_profile": "activation_heavy",
    },
    "bromothiophene_screen": {
        "substrate": "bromothiophene electrophile",
        "product": "thiophene arylamine",
        "substrate_features": (
            "aryl_bromide",
            "heteroaryl",
            "sulfur_heteroaryl",
        ),
        "product_features": ("arylamine", "heteroaryl_product"),
        "known_incompatibilities": (
            "excessively polar solvents with base-sensitive sulfur motifs",
        ),
        "constraints": (
            "avoid obviously unstable extremes",
            "retain exploratory solvent diversity",
        ),
        "desired_axes": ("ligand", "base", "solvent", "temperature"),
        "risk_profile": "balanced",
    },
}


def build_task_representation(profile_name: str, reaction_family: str, objective: str) -> TaskRepresentation:
    if profile_name not in _PROFILE_LIBRARY:
        raise KeyError(f"Unknown profile: {profile_name}")
    profile = _PROFILE_LIBRARY[profile_name]
    return TaskRepresentation(
        profile_name=profile_name,
        reaction_family=reaction_family,
        objective=objective,
        substrate=str(profile["substrate"]),
        product=str(profile["product"]),
        substrate_features=tuple(profile["substrate_features"]),
        product_features=tuple(profile["product_features"]),
        known_incompatibilities=tuple(profile["known_incompatibilities"]),
        constraints=tuple(profile["constraints"]),
        desired_axes=tuple(profile["desired_axes"]),
        risk_profile=str(profile["risk_profile"]),
        metadata={"profile_name": profile_name},
    )


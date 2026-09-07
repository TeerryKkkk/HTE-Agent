from __future__ import annotations

from pipeline_v2.shared.candidate_schema import Candidate, Observation, TrustRegion
from pipeline_v2.shared.io_utils import bounded, mean


def build_trust_region(observations: list[Observation]) -> TrustRegion:
    if not observations:
        return TrustRegion(
            preferred_values={},
            temperature_window=(50.0, 110.0),
            catalyst_window=(0.10, 0.20),
            base_equiv_window=(1.5, 3.0),
            notes=["no_observations_yet"],
        )
    mean_yield = mean([item.yield_value for item in observations])
    anchors = [item for item in observations if item.yield_value >= mean_yield]
    if not anchors:
        anchors = sorted(observations, key=lambda item: item.yield_value, reverse=True)[: max(3, len(observations) // 3)]

    preferred_values = {
        "ligand_label": sorted({str(item.parameters.get("ligand_label", "")) for item in anchors}),
        "base_label": sorted({str(item.parameters.get("base_label", "")) for item in anchors}),
        "solvent_label": sorted({str(item.parameters.get("solvent_label", "")) for item in anchors}),
    }
    temperatures = [float(item.parameters.get("temperature_c", 80.0)) for item in anchors]
    catalyst_values = [float(item.parameters.get("catalyst_mol_pct", 0.15)) for item in anchors]
    base_equivs = [float(item.parameters.get("base_equiv", 2.0)) for item in anchors]
    return TrustRegion(
        preferred_values=preferred_values,
        temperature_window=(bounded(min(temperatures) - 8.0, 50.0, 110.0), bounded(max(temperatures) + 8.0, 50.0, 110.0)),
        catalyst_window=(bounded(min(catalyst_values) - 0.03, 0.08, 0.22), bounded(max(catalyst_values) + 0.03, 0.08, 0.22)),
        base_equiv_window=(bounded(min(base_equivs) - 0.4, 1.0, 3.2), bounded(max(base_equivs) + 0.4, 1.0, 3.2)),
        notes=[f"anchor_count={len(anchors)}", f"mean_yield={mean_yield:.3f}"],
    )


def trust_region_bonus(candidate: Candidate, trust_region: TrustRegion) -> float:
    params = candidate.parameters
    bonus = 0.0
    for axis, weight in (("ligand_label", 0.03), ("base_label", 0.02), ("solvent_label", 0.02)):
        if str(params.get(axis, "")) in set(trust_region.preferred_values.get(axis, [])):
            bonus += weight
    temperature = float(params.get("temperature_c", 80.0))
    catalyst_mol_pct = float(params.get("catalyst_mol_pct", 0.15))
    base_equiv = float(params.get("base_equiv", 2.0))
    if trust_region.temperature_window[0] <= temperature <= trust_region.temperature_window[1]:
        bonus += 0.02
    if trust_region.catalyst_window[0] <= catalyst_mol_pct <= trust_region.catalyst_window[1]:
        bonus += 0.01
    if trust_region.base_equiv_window[0] <= base_equiv <= trust_region.base_equiv_window[1]:
        bonus += 0.01
    return round(bonus, 4)

from __future__ import annotations

from math import sqrt

from hte_agent.shared.candidate_schema import Candidate, Observation, Stage2Score, TrustRegion
from hte_agent.shared.io_utils import bounded, safe_divide
from hte_agent.stage2.trust_region import trust_region_bonus


class MixedBatchOptimizer:
    """Mixed-space batch search using similarity-weighted local regression.

    The uncertainty score combines local variance and novelty; it is not a
    Bayesian posterior standard deviation. Diversity and trust-region terms
    are deterministic selection heuristics.
    """

    def __init__(self, acquisition_beta: float, diversity_weight: float, prior_blend: float) -> None:
        self.acquisition_beta = acquisition_beta
        self.diversity_weight = diversity_weight
        self.prior_blend = prior_blend

    def _similarity(self, left: dict[str, object], right: dict[str, object]) -> float:
        score = 0.0
        weight = 0.0
        for key, current_weight in (("ligand_label", 1.5), ("base_label", 1.4), ("solvent_label", 1.2)):
            score += current_weight * (1.0 if str(left.get(key, "")) == str(right.get(key, "")) else 0.0)
            weight += current_weight
        for key, span in (("temperature_c", 40.0), ("catalyst_mol_pct", 0.1), ("base_equiv", 1.5)):
            gap = abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
            score += max(0.0, 1.0 - gap / span)
            weight += 1.0
        return safe_divide(score, weight)

    def _predict(self, candidate: Candidate, observations: list[Observation], prior_score: float) -> tuple[float, float]:
        if not observations:
            return prior_score, 0.18
        similarities = [self._similarity(candidate.parameters, observation.parameters) for observation in observations]
        weights = [max(0.02, similarity**2) for similarity in similarities]
        weighted_sum = sum(weight * observation.yield_value for weight, observation in zip(weights, observations))
        local_mean = safe_divide(weighted_sum, sum(weights))
        local_var = safe_divide(
            sum(weight * (observation.yield_value - local_mean) ** 2 for weight, observation in zip(weights, observations)),
            sum(weights),
        )
        novelty = 1.0 - max(similarities)
        blended_mean = (1.0 - self.prior_blend) * local_mean + self.prior_blend * prior_score
        predicted_std = bounded(sqrt(local_var) + 0.16 * novelty + 0.04, 0.05, 0.28)
        return round(blended_mean, 4), round(predicted_std, 4)

    def rank_candidates(
        self,
        candidates: list[Candidate],
        observations: list[Observation],
        trust_region: TrustRegion,
        observed_ids: set[str],
    ) -> list[Stage2Score]:
        ranked: list[Stage2Score] = []
        for candidate in candidates:
            if candidate.candidate_id in observed_ids or not candidate.feasible:
                continue
            prior_score = candidate.stage1_score
            predicted_mean, predicted_std = self._predict(candidate, observations, prior_score)
            region_bonus = trust_region_bonus(candidate, trust_region)
            acquisition_score = predicted_mean + self.acquisition_beta * predicted_std + region_bonus
            ranked.append(
                Stage2Score(
                    candidate_id=candidate.candidate_id,
                    predicted_mean=predicted_mean,
                    predicted_std=predicted_std,
                    acquisition_score=round(acquisition_score, 4),
                    prior_score=round(prior_score, 4),
                    trust_region_bonus=region_bonus,
                    feasibility_penalty=0.0,
                    final_score=round(acquisition_score, 4),
                    components={
                        "predicted_mean": predicted_mean,
                        "predicted_std": predicted_std,
                        "trust_region_bonus": region_bonus,
                        "prior_score": round(prior_score, 4),
                    },
                )
            )
        ranked.sort(key=lambda item: item.final_score, reverse=True)
        return ranked

    def select_diverse_batch(
        self,
        scored_candidates: list[Stage2Score],
        candidate_by_id: dict[str, Candidate],
        batch_size: int,
    ) -> tuple[list[Candidate], list[dict[str, object]]]:
        remaining = {score.candidate_id: score for score in scored_candidates}
        selected: list[Candidate] = []
        trace: list[dict[str, object]] = []
        while remaining and len(selected) < batch_size:
            best_id = ""
            best_value = -999.0
            best_diversity_penalty = 0.0
            for candidate_id, score in remaining.items():
                candidate = candidate_by_id[candidate_id]
                diversity_penalty = 0.0
                for chosen in selected:
                    diversity_penalty = max(diversity_penalty, self._similarity(candidate.parameters, chosen.parameters))
                selection_value = score.final_score - self.diversity_weight * diversity_penalty
                if selection_value > best_value:
                    best_id = candidate_id
                    best_value = selection_value
                    best_diversity_penalty = diversity_penalty
            chosen_score = remaining.pop(best_id)
            chosen_candidate = candidate_by_id[best_id]
            selected.append(chosen_candidate)
            trace.append(
                {
                    "slot": len(selected),
                    "candidate_id": best_id,
                    "final_score": chosen_score.final_score,
                    "diversity_penalty": round(self.diversity_weight * best_diversity_penalty, 4),
                    "selection_value": round(best_value, 4),
                    "llm_decision": chosen_score.llm_decision,
                }
            )
        return selected, trace

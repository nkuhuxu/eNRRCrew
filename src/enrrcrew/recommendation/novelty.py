from __future__ import annotations

from enrrcrew.predictors.common import categorize_structure
from enrrcrew.schemas import RecommendationCandidate


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left.union(right)
    return 1.0 if not union else len(left.intersection(right)) / len(union)


def candidate_similarity(
    candidate: RecommendationCandidate, known: RecommendationCandidate
) -> float:
    element_similarity = _jaccard(set(candidate.elements), set(known.elements))
    morphology_similarity = float(
        (categorize_structure(candidate.morphology) or "")
        == (categorize_structure(known.morphology) or "")
    )
    ph_similarity = _jaccard(candidate.ph_categories, known.ph_categories)
    electrolyte_similarity = _jaccard(candidate.electrolytes, known.electrolytes)
    potential_similarity = max(
        0.0, 1.0 - abs(candidate.applied_potential - known.applied_potential) / 0.5
    )
    return (
        0.55 * element_similarity
        + 0.15 * morphology_similarity
        + 0.10 * ph_similarity
        + 0.10 * electrolyte_similarity
        + 0.10 * potential_similarity
    )


def assess_novelty(
    candidate: RecommendationCandidate,
    known_candidates: list[RecommendationCandidate],
    limit: int = 3,
) -> tuple[float, list[dict[str, object]]]:
    similarities = [
        (candidate_similarity(candidate, known), known) for known in known_candidates
    ]
    similarities.sort(key=lambda item: (-item[0], item[1].candidate_id))
    nearest = [
        {
            "candidate_id": known.candidate_id,
            "name": known.display_name,
            "similarity": round(score, 6),
            "elements": known.elements,
            "morphology": known.morphology,
            "applied_potential": known.applied_potential,
        }
        for score, known in similarities[:limit]
    ]
    maximum = similarities[0][0] if similarities else 0.0
    return max(0.0, min(1.0, 1.0 - maximum)), nearest

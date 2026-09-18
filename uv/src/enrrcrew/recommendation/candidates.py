from __future__ import annotations

from collections.abc import Iterable

from enrrcrew.predictors.common import STRUCTURE_TYPES, categorize_structure
from enrrcrew.recommendation.catalog import (
    CatalystCatalog,
    candidate_id,
    candidate_signature,
)
from enrrcrew.schemas import (
    CandidateOrigin,
    RecommendationCandidate,
    RecommendationRequest,
)


def filter_known_candidates(
    catalog: CatalystCatalog, request: RecommendationRequest
) -> list[RecommendationCandidate]:
    requested_morphologies = {
        categorize_structure(value) or "" for value in request.morphologies
    }
    selected = []
    for candidate in catalog.candidates:
        elements = set(candidate.elements)
        if not elements.issubset(request.allowed_elements):
            continue
        if elements.intersection(request.forbidden_elements):
            continue
        if len(elements) > request.max_elements:
            continue
        if not request.potential_min <= candidate.applied_potential <= request.potential_max:
            continue
        morphology = categorize_structure(candidate.morphology) or ""
        if requested_morphologies and morphology not in requested_morphologies:
            continue
        if request.ph_categories and not candidate.ph_categories.intersection(
            request.ph_categories
        ):
            continue
        if request.electrolytes and not candidate.electrolytes.intersection(
            request.electrolytes
        ):
            continue
        selected.append(candidate)
    return selected


def _support(
    element: str, existing: Iterable[str], cooccurrence: dict[tuple[str, str], int]
) -> int:
    return max(
        (
            cooccurrence.get(tuple(sorted((element, other))), 0)
            for other in existing
            if other != element
        ),
        default=0,
    )


def _generated_candidate(
    seed: RecommendationCandidate,
    elements: set[str],
    morphology: str,
    potential: float,
) -> RecommendationCandidate:
    ordered_elements = sorted(elements)
    display_name = f"{'–'.join(ordered_elements)} / {morphology or 'unspecified morphology'} hypothesis"
    provisional = RecommendationCandidate(
        candidate_id="pending",
        display_name=display_name,
        origin=CandidateOrigin.GENERATED,
        elements=ordered_elements,
        morphology=morphology,
        applied_potential=potential,
        ph_categories=seed.ph_categories,
        electrolytes=seed.electrolytes,
    )
    signature = candidate_signature(provisional)
    return provisional.model_copy(update={"candidate_id": candidate_id(signature)})


def generate_candidates(
    seeds: list[RecommendationCandidate],
    catalog: CatalystCatalog,
    request: RecommendationRequest,
) -> list[RecommendationCandidate]:
    if request.generated_candidate_limit == 0:
        return []
    known_signatures = catalog.signatures
    morphologies = sorted(
        {categorize_structure(value) or "" for value in request.morphologies}
        or set(STRUCTURE_TYPES)
    )
    potentials = request.potential_grid()
    allowed = sorted(request.allowed_elements.difference(request.forbidden_elements))
    proposals: list[tuple[int, str, RecommendationCandidate]] = []
    seen: set[str] = set()

    for seed in seeds:
        seed_elements = set(seed.elements)
        element_variants: list[tuple[int, set[str]]] = []
        if len(seed_elements) < request.max_elements:
            for element in allowed:
                if element in seed_elements:
                    continue
                support = _support(element, seed_elements, catalog.cooccurrence)
                if support >= 3:
                    element_variants.append((support, seed_elements | {element}))
        if len(seed_elements) > 1:
            for element in sorted(seed_elements):
                element_variants.append((0, seed_elements - {element}))
        for removed in sorted(seed_elements):
            remaining = seed_elements - {removed}
            if not remaining:
                continue
            for added in allowed:
                if added in seed_elements:
                    continue
                support = _support(added, remaining, catalog.cooccurrence)
                if support >= 3:
                    element_variants.append((support, remaining | {added}))

        structural_variants = [(0, seed_elements, morphology) for morphology in morphologies]
        variants = [
            (support, elements, seed.morphology) for support, elements in element_variants
        ] + structural_variants
        for support, elements, morphology in variants:
            if not elements or len(elements) > request.max_elements:
                continue
            for potential in potentials:
                candidate = _generated_candidate(seed, elements, morphology, potential)
                signature = candidate_signature(candidate)
                if signature in known_signatures or signature in seen:
                    continue
                seen.add(signature)
                proposals.append((-support, signature, candidate))

    proposals.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in proposals[: request.generated_candidate_limit]]

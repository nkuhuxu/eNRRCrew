from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from enrrcrew.recommendation.applicability import ApplicabilityDomain
from enrrcrew.recommendation.candidates import filter_known_candidates, generate_candidates
from enrrcrew.recommendation.catalog import (
    CatalystCatalog,
    candidate_signature,
    load_catalog,
)
from enrrcrew.recommendation.novelty import assess_novelty
from enrrcrew.schemas import (
    CandidateOrigin,
    RecommendationCandidate,
    RecommendationRequest,
)


def candidate(
    identifier: str,
    elements: list[str],
    *,
    morphology: str = "Nanosheets",
    potential: float = -0.3,
    origin: CandidateOrigin = CandidateOrigin.KNOWN,
) -> RecommendationCandidate:
    return RecommendationCandidate(
        candidate_id=identifier,
        display_name=identifier,
        origin=origin,
        elements=elements,
        morphology=morphology,
        applied_potential=potential,
        ph_categories={"neutral"},
        electrolytes={"pbs"},
        source_rows=[0] if origin is CandidateOrigin.KNOWN else [],
    )


def request(**updates: object) -> RecommendationRequest:
    values = {
        "mode": "hybrid",
        "allowed_elements": {"Fe", "Mo", "N", "C"},
        "potential_min": -0.5,
        "potential_max": -0.1,
        "potential_step": 0.1,
        "generated_candidate_limit": 20,
        "result_limit": 5,
    }
    values.update(updates)
    return RecommendationRequest.model_validate(values)


def test_recommendation_request_rejects_overlap_and_large_grid() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        request(forbidden_elements={"Fe"})
    with pytest.raises(ValidationError, match="at most 25"):
        request(potential_min=-2.0, potential_max=0.0, potential_step=0.05)


def test_catalog_uses_both_potentials_and_deduplicates(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "Electrocatalyst": "Fe-N-C",
                "Elements of electrocatalyst_0": "Fe",
                "Elements of electrocatalyst_1": "N",
                "Elements of electrocatalyst_2": "C",
                "Morphology of electrocatalyst": "porous nanosheet",
                "Applied Potential (Faraday Efficiency)": -0.2,
                "Applied Potential (NH3 Yield)": -0.3,
                "pH_neutral": 1,
                "Electrolyte without concentration_pbs": 1,
            },
            {
                "Electrocatalyst": "Fe-N-C duplicate",
                "Elements of electrocatalyst_0": "Fe",
                "Elements of electrocatalyst_1": "N",
                "Elements of electrocatalyst_2": "C",
                "Morphology of electrocatalyst": "porous nanosheet",
                "Applied Potential (Faraday Efficiency)": -0.2,
                "Applied Potential (NH3 Yield)": -0.3,
                "pH_neutral": 1,
                "Electrolyte without concentration_pbs": 1,
            },
        ]
    )
    path = tmp_path / "catalog.csv"
    frame.to_csv(path, index=False)

    catalog = load_catalog(str(path))

    assert len(catalog.candidates) == 2
    assert {item.applied_potential for item in catalog.candidates} == {-0.2, -0.3}
    assert all(item.source_rows == [0, 1] for item in catalog.candidates)


def test_generation_is_deterministic_and_respects_constraints() -> None:
    seed = candidate("seed", ["Fe", "N", "C"])
    catalog = CatalystCatalog(
        candidates=(seed,),
        element_counts=Counter({"Fe": 5, "N": 5, "C": 5, "Mo": 5}),
        cooccurrence={("Fe", "Mo"): 4, ("Mo", "N"): 3},
    )
    configured = request(morphologies={"Nanosheets"}, max_elements=4)

    first = generate_candidates([seed], catalog, configured)
    second = generate_candidates([seed], catalog, configured)

    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert all(set(item.elements).issubset(configured.allowed_elements) for item in first)
    assert all(len(item.elements) <= 4 for item in first)
    assert any("Mo" in item.elements for item in first)


def test_generation_can_be_disabled() -> None:
    seed = candidate("seed", ["Fe", "N", "C"])
    catalog = CatalystCatalog(
        candidates=(seed,), element_counts=Counter(), cooccurrence={}
    )
    assert generate_candidates(
        [seed], catalog, request(generated_candidate_limit=0)
    ) == []


@pytest.mark.parametrize(
    "updates",
    [
        {"allowed_elements": {"Mo"}},
        {"max_elements": 2},
        {"potential_min": -0.2},
        {"morphologies": {"Nanoparticles"}},
        {"ph_categories": {"acidic"}},
        {"electrolytes": {"koh"}},
    ],
)
def test_known_candidate_filters_each_constraint(updates: dict[str, object]) -> None:
    known = candidate("known", ["Fe", "N", "C"])
    catalog = CatalystCatalog(
        candidates=(known,), element_counts=Counter(), cooccurrence={}
    )
    assert filter_known_candidates(catalog, request(**updates)) == []


def test_filter_and_novelty_use_local_signature() -> None:
    known = candidate("known", ["Fe", "N", "C"])
    catalog = CatalystCatalog(
        candidates=(known,), element_counts=Counter(), cooccurrence={}
    )
    assert filter_known_candidates(catalog, request()) == [known]
    novelty, nearest = assess_novelty(known, [known])
    assert novelty == pytest.approx(0.0)
    assert nearest[0]["candidate_id"] == "known"
    assert candidate_signature(known).endswith("-0.300000")


def test_applicability_marks_far_vectors_out_of_domain() -> None:
    reference = np.asarray([[0.0], [0.1], [0.2], [0.3], [0.4]])
    domain = ApplicabilityDomain.from_reference(reference, reference)

    near, far = domain.assess(np.asarray([[0.15], [10.0]]), np.asarray([[0.15], [10.0]]))

    assert near[1] == "in_domain"
    assert far[1] == "out_of_domain"

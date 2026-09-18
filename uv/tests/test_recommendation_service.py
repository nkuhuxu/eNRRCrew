from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from enrrcrew.recommendation import RecommendationService
from enrrcrew.recommendation import service as service_module
from enrrcrew.schemas import (
    CandidateOrigin,
    PredictionInput,
    PredictionResult,
    RecommendationCandidate,
    RecommendationRequest,
)


class FakeYieldPredictor:
    def evaluate_many(
        self, values: list[PredictionInput]
    ) -> tuple[list[PredictionResult], np.ndarray]:
        return self.predict_many(values), np.asarray(
            [[value.applied_potential, len(value.elements)] for value in values], dtype=float
        )

    def predict_many(self, values: list[PredictionInput]) -> list[PredictionResult]:
        return [
            PredictionResult(
                prediction_type=value.prediction_type,
                electrocatalyst=value.electrocatalyst,
                category="High",
                probability=0.9,
                model_version="yield:test",
                input_data=value.model_dump(mode="json"),
            )
            for value in values
        ]


class FakeFEPredictor:
    def evaluate_many(
        self, values: list[PredictionInput]
    ) -> tuple[list[PredictionResult], np.ndarray, list[float]]:
        return (
            self.predict_many(values),
            np.asarray(
                [[value.applied_potential, len(value.elements)] for value in values],
                dtype=float,
            ),
            [0.6 for _ in values],
        )

    def predict_many(self, values: list[PredictionInput]) -> list[PredictionResult]:
        return [
            PredictionResult(
                prediction_type=value.prediction_type,
                electrocatalyst=value.electrocatalyst,
                category="High",
                cluster=3,
                model_version="fe:test",
                input_data=value.model_dump(mode="json"),
            )
            for value in values
        ]


def dataset(path: Path) -> None:
    pd.DataFrame(
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
            }
        ]
    ).to_csv(path, index=False)


def test_service_returns_only_double_high_and_persists_audit(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    dataset(csv_path)
    session_root = tmp_path / "session"
    service = RecommendationService(csv_path, tmp_path, session_root)
    service.yield_predictor = FakeYieldPredictor()
    service.fe_predictor = FakeFEPredictor()
    request = RecommendationRequest(
        mode="known",
        allowed_elements={"Fe", "N", "C"},
        potential_min=-0.3,
        potential_max=-0.2,
        potential_step=0.1,
        generated_candidate_limit=0,
        result_limit=10,
    )

    result = service.recommend(request)

    assert len(result.recommendations) == 2
    assert not result.near_misses
    assert all(item.yield_result.category == "High" for item in result.recommendations)
    assert all(item.fe_result.category == "High" for item in result.recommendations)
    run_dir = session_root / "recommendations" / result.run_id
    assert (run_dir / "request.json").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "recommendations.csv").is_file()
    assert "api_key" not in (run_dir / "result.json").read_text(encoding="utf-8")


def test_service_rejects_empty_constraint_result(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    dataset(csv_path)
    service = RecommendationService(csv_path, tmp_path)
    service.yield_predictor = FakeYieldPredictor()
    service.fe_predictor = FakeFEPredictor()
    request = RecommendationRequest(
        mode="known",
        allowed_elements={"Mo"},
        generated_candidate_limit=0,
    )
    with pytest.raises(ValueError, match="No historical candidates"):
        service.recommend(request)


def test_missing_metadata_domain_and_n15_warnings(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    dataset(csv_path)
    service = RecommendationService(csv_path, tmp_path)
    service.yield_predictor = FakeYieldPredictor()
    service.fe_predictor = FakeFEPredictor()
    incomplete = RecommendationCandidate(
        candidate_id="generated",
        display_name="Fe hypothesis",
        origin=CandidateOrigin.GENERATED,
        elements=["Fe"],
        morphology="",
        applied_potential=-0.3,
        ph_categories=set(),
        electrolytes=set(),
    )

    class BorderlineDomain:
        @staticmethod
        def assess(yield_vectors, fe_vectors):
            return [(0.4, "borderline")]

    items, _, _ = service._evaluate([incomplete], BorderlineDomain(), [incomplete])
    assert {
        "Historical morphology is missing.",
        "Historical pH category is missing.",
        "Historical electrolyte category is missing.",
        "Applicability domain: borderline.",
    }.issubset(items[0].warnings)
    assert service._n15_sensitivity([]) == []

    class ChangedYield(FakeYieldPredictor):
        def predict_many(self, values):
            changed = super().predict_many(values)
            return [item.model_copy(update={"category": "Low"}) for item in changed]

    service.yield_predictor = ChangedYield()
    updated = service._n15_sensitivity(items)
    assert any("N-15 labeling" in warning for warning in updated[0].warnings)


@pytest.mark.parametrize("mode", ["explore", "hybrid"])
def test_generated_modes_use_generated_candidates(
    monkeypatch, tmp_path: Path, mode: str
) -> None:
    csv_path = tmp_path / "dataset.csv"
    dataset(csv_path)
    service = RecommendationService(csv_path, tmp_path)
    service.yield_predictor = FakeYieldPredictor()
    service.fe_predictor = FakeFEPredictor()
    generated = RecommendationCandidate(
        candidate_id="generated",
        display_name="Fe-Mo hypothesis",
        origin=CandidateOrigin.GENERATED,
        elements=["Fe", "Mo"],
        morphology="Nanosheets",
        applied_potential=-0.25,
        ph_categories={"neutral"},
        electrolytes={"pbs"},
    )
    monkeypatch.setattr(
        service_module, "generate_candidates", lambda seeds, catalog, request: [generated]
    )
    request = RecommendationRequest(
        mode=mode,
        allowed_elements={"Fe", "Mo", "N", "C"},
        potential_min=-0.3,
        potential_max=-0.2,
        generated_candidate_limit=1,
        result_limit=10,
    )
    result = service.recommend(request)
    assert result.rejected_counts["generated"] == 1
    if mode == "explore":
        assert all(item.candidate.origin is CandidateOrigin.GENERATED for item in result.recommendations)

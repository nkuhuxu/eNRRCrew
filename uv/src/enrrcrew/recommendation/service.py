from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from enrrcrew.predictors import FEPredictor, YieldPredictor
from enrrcrew.recommendation.applicability import ApplicabilityDomain
from enrrcrew.recommendation.candidates import filter_known_candidates, generate_candidates
from enrrcrew.recommendation.catalog import CatalystCatalog, load_catalog
from enrrcrew.recommendation.novelty import assess_novelty
from enrrcrew.schemas import (
    CandidateOrigin,
    PredictionInput,
    PredictionType,
    RecommendationCandidate,
    RecommendationItem,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
)


def _prediction_input(
    candidate: RecommendationCandidate, prediction_type: PredictionType, n15: bool = False
) -> PredictionInput:
    return PredictionInput(
        prediction_type=prediction_type,
        applied_potential=candidate.applied_potential,
        electrocatalyst=candidate.display_name,
        elements=candidate.elements,
        morphology=candidate.morphology,
        ph_categories=candidate.ph_categories,
        electrolytes=candidate.electrolytes,
        n15_labeling=n15,
    )


def _rank_key(item: RecommendationItem) -> tuple[int, float, float, str]:
    domain_rank = {"in_domain": 0, "borderline": 1, "out_of_domain": 2}
    return (
        domain_rank[item.applicability_status],
        -item.combined_signal,
        -item.novelty_score,
        item.candidate.candidate_id,
    )


class RecommendationService:
    def __init__(
        self, dataset_path: Path, models_dir: Path, session_root: Path | None = None
    ) -> None:
        self.dataset_path = dataset_path.resolve()
        self.models_dir = models_dir.resolve()
        self.session_root = session_root.resolve() if session_root else None
        self.catalog: CatalystCatalog = load_catalog(str(self.dataset_path))
        self.yield_predictor = YieldPredictor(self.models_dir)
        self.fe_predictor = FEPredictor(self.models_dir)
        self._reference_cache: tuple[
            list[RecommendationItem], np.ndarray, np.ndarray, ApplicabilityDomain
        ] | None = None

    def default_allowed_elements(self) -> list[str]:
        return self.catalog.default_allowed_elements()

    def _evaluate(
        self,
        candidates: list[RecommendationCandidate],
        domain: ApplicabilityDomain | None = None,
        known_for_novelty: list[RecommendationCandidate] | None = None,
        compute_novelty: bool = True,
    ) -> tuple[list[RecommendationItem], np.ndarray, np.ndarray]:
        yield_inputs = [
            _prediction_input(candidate, PredictionType.YIELD) for candidate in candidates
        ]
        fe_inputs = [_prediction_input(candidate, PredictionType.FE) for candidate in candidates]
        yield_results, yield_vectors = self.yield_predictor.evaluate_many(yield_inputs)
        fe_results, fe_vectors, margins = self.fe_predictor.evaluate_many(fe_inputs)
        assessments = (
            domain.assess(yield_vectors, fe_vectors)
            if domain
            else [(1.0, "in_domain") for _ in candidates]
        )
        novelty_reference = known_for_novelty if known_for_novelty is not None else candidates
        items = []
        for index, candidate in enumerate(candidates):
            novelty, nearest = (
                assess_novelty(candidate, novelty_reference)
                if compute_novelty
                else (0.0, [])
            )
            if candidate.origin is CandidateOrigin.KNOWN:
                novelty = 0.0
            probability = yield_results[index].probability
            yield_score = probability if probability is not None else 0.0
            fe_signal = max(margins[index], 0.0)
            combined = math.sqrt(yield_score * fe_signal)
            warnings = list(
                dict.fromkeys(yield_results[index].warnings + fe_results[index].warnings)
            )
            if not candidate.morphology:
                warnings.append("Historical morphology is missing.")
            if not candidate.ph_categories:
                warnings.append("Historical pH category is missing.")
            if not candidate.electrolytes:
                warnings.append("Historical electrolyte category is missing.")
            applicability_score, applicability_status = assessments[index]
            if applicability_status != "in_domain":
                warnings.append(f"Applicability domain: {applicability_status}.")
            items.append(
                RecommendationItem(
                    candidate=candidate,
                    yield_result=yield_results[index],
                    fe_result=fe_results[index],
                    yield_high_score=probability,
                    fe_centroid_margin=margins[index],
                    combined_signal=combined,
                    applicability_score=applicability_score,
                    applicability_status=applicability_status,
                    novelty_score=novelty,
                    nearest_known_candidates=nearest,
                    warnings=warnings,
                )
            )
        return items, yield_vectors, fe_vectors

    def _reference_data(
        self,
    ) -> tuple[list[RecommendationItem], np.ndarray, np.ndarray, ApplicabilityDomain]:
        if self._reference_cache is None:
            candidates = list(self.catalog.candidates)
            items, yield_vectors, fe_vectors = self._evaluate(
                candidates, compute_novelty=False
            )
            domain = ApplicabilityDomain.from_reference(yield_vectors, fe_vectors)
            self._reference_cache = (items, yield_vectors, fe_vectors, domain)
        return self._reference_cache

    @staticmethod
    def _strict(item: RecommendationItem) -> bool:
        return (
            item.yield_result.category == "High"
            and item.fe_result.category == "High"
            and item.applicability_status != "out_of_domain"
        )

    @staticmethod
    def _apply_domain(
        items: list[RecommendationItem],
        yield_vectors: np.ndarray,
        fe_vectors: np.ndarray,
        domain: ApplicabilityDomain,
    ) -> list[RecommendationItem]:
        updated = []
        for item, (score, status) in zip(
            items, domain.assess(yield_vectors, fe_vectors), strict=True
        ):
            warnings = [
                warning
                for warning in item.warnings
                if not warning.startswith("Applicability domain:")
            ]
            if status != "in_domain":
                warnings.append(f"Applicability domain: {status}.")
            updated.append(
                item.model_copy(
                    update={
                        "applicability_score": score,
                        "applicability_status": status,
                        "warnings": warnings,
                    }
                )
            )
        return updated

    def _n15_sensitivity(self, items: list[RecommendationItem]) -> list[RecommendationItem]:
        if not items:
            return items
        candidates = [item.candidate for item in items]
        yield_results = self.yield_predictor.predict_many(
            [_prediction_input(candidate, PredictionType.YIELD, True) for candidate in candidates]
        )
        fe_results = self.fe_predictor.predict_many(
            [_prediction_input(candidate, PredictionType.FE, True) for candidate in candidates]
        )
        updated = []
        for item, yield_result, fe_result in zip(items, yield_results, fe_results, strict=True):
            warnings = list(item.warnings)
            if (
                yield_result.category != item.yield_result.category
                or fe_result.category != item.fe_result.category
            ):
                warnings.append(
                    "Prediction changes when N-15 labeling is toggled; treat this as reporting-bias sensitivity."
                )
            updated.append(item.model_copy(update={"warnings": warnings}))
        return updated

    def _with_nearest_known(
        self, items: list[RecommendationItem]
    ) -> list[RecommendationItem]:
        known = list(self.catalog.candidates)
        updated = []
        for item in items:
            if item.nearest_known_candidates:
                updated.append(item)
                continue
            _, nearest = assess_novelty(item.candidate, known)
            updated.append(item.model_copy(update={"nearest_known_candidates": nearest}))
        return updated

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        known_candidates = filter_known_candidates(self.catalog, request)
        if not known_candidates:
            raise ValueError("No historical candidates match the requested constraints")

        reference_items, yield_reference, fe_reference, domain = self._reference_data()
        reference_index = {
            item.candidate.candidate_id: index for index, item in enumerate(reference_items)
        }
        indexes = [reference_index[candidate.candidate_id] for candidate in known_candidates]
        known_items = [reference_items[index] for index in indexes]
        known_items = self._apply_domain(
            known_items, yield_reference[indexes], fe_reference[indexes], domain
        )
        ranked_known = sorted(known_items, key=_rank_key)
        strict_seeds = [item for item in ranked_known if self._strict(item)]
        seed_items = (strict_seeds or ranked_known)[:50]

        generated_candidates = []
        generated_items: list[RecommendationItem] = []
        if request.mode in {RecommendationMode.EXPLORE, RecommendationMode.HYBRID}:
            generated_candidates = generate_candidates(
                [item.candidate for item in seed_items], self.catalog, request
            )
            if generated_candidates:
                generated_items, _, _ = self._evaluate(
                    generated_candidates, domain, known_candidates
                )

        if request.mode is RecommendationMode.KNOWN:
            pool = known_items
        elif request.mode is RecommendationMode.EXPLORE:
            pool = generated_items
        else:
            pool = known_items + generated_items

        recommendations = sorted(
            (item for item in pool if self._strict(item)), key=_rank_key
        )[: request.result_limit]
        near_misses = sorted(
            (item for item in pool if not self._strict(item)), key=_rank_key
        )[: request.result_limit]
        recommendations = self._n15_sensitivity(
            self._with_nearest_known(recommendations)
        )
        near_misses = self._with_nearest_known(near_misses)
        rejected_counts = {
            "evaluated_known": len(known_items),
            "generated": len(generated_items),
            "yield_low": sum(item.yield_result.category != "High" for item in pool),
            "fe_low": sum(item.fe_result.category != "High" for item in pool),
            "out_of_domain": sum(
                item.applicability_status == "out_of_domain" for item in pool
            ),
            "strict_double_high": sum(self._strict(item) for item in pool),
        }
        first = (recommendations or near_misses or pool or reference_items)[0]
        result = RecommendationResult(
            request=request,
            recommendations=recommendations,
            near_misses=near_misses,
            rejected_counts=rejected_counts,
            model_versions={
                "yield": first.yield_result.model_version,
                "fe": first.fe_result.model_version,
            },
            run_id=uuid4().hex,
        )
        self._persist(result)
        return result

    def _persist(self, result: RecommendationResult) -> None:
        if self.session_root is None:
            return
        run_dir = self.session_root / "recommendations" / result.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "request.json").write_text(
            result.request.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        rows = []
        for item in result.recommendations:
            rows.append(
                {
                    "candidate_id": item.candidate.candidate_id,
                    "name": item.candidate.display_name,
                    "origin": item.candidate.origin.value,
                    "elements": ",".join(item.candidate.elements),
                    "morphology": item.candidate.morphology,
                    "applied_potential": item.candidate.applied_potential,
                    "ph_categories": json.dumps(sorted(item.candidate.ph_categories)),
                    "electrolytes": json.dumps(sorted(item.candidate.electrolytes)),
                    "yield_category": item.yield_result.category,
                    "yield_high_score": item.yield_high_score,
                    "fe_category": item.fe_result.category,
                    "fe_centroid_margin": item.fe_centroid_margin,
                    "combined_signal": item.combined_signal,
                    "applicability_status": item.applicability_status,
                    "applicability_score": item.applicability_score,
                    "novelty_score": item.novelty_score,
                    "warnings": " | ".join(item.warnings),
                }
            )
        pd.DataFrame(rows).to_csv(run_dir / "recommendations.csv", index=False)

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

from enrrcrew.predictors.common import build_feature_frames, file_version
from enrrcrew.schemas import PredictionInput, PredictionResult, PredictionType


@lru_cache(maxsize=4)
def _load_resources(models_dir_text: str) -> tuple[object, object, list[str], list[str], object, object]:
    models_dir = Path(models_dir_text)
    all_features = list(load(models_dir / "features_for_yield_before_RFE.joblib"))
    selected_features = [
        line.strip()
        for line in (models_dir / "selected_features_for_yield.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return (
        load(models_dir / "scaler_for_yield.joblib"),
        load(models_dir / "best_model_for_yield.joblib"),
        all_features,
        selected_features,
        StrToComposition(),
        ElementProperty.from_preset(preset_name="magpie"),
    )


class YieldPredictor:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir.resolve()

    def predict(self, value: PredictionInput) -> PredictionResult:
        return self.predict_many([value])[0]

    def evaluate_many(
        self, values: list[PredictionInput]
    ) -> tuple[list[PredictionResult], np.ndarray]:
        if any(value.prediction_type is not PredictionType.YIELD for value in values):
            raise ValueError("YieldPredictor requires prediction_type='yield'")
        if not values:
            return [], np.empty((0, 0))
        scaler, model, all_features, selected, converter, element_property = _load_resources(
            str(self.models_dir)
        )
        features, warnings = build_feature_frames(
            values, all_features, converter, element_property
        )
        scaled = scaler.transform(features)
        scaled_frame = pd.DataFrame(scaled, columns=all_features)
        missing_selected = [name for name in selected if name not in scaled_frame.columns]
        if missing_selected:
            raise ValueError(f"selected yield features are unavailable: {missing_selected}")
        selected_frame = scaled_frame[selected]
        model_input = selected_frame.to_numpy()
        raw_predictions = model.predict(model_input)
        probability_by_row: list[float | None] = [None] * len(values)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)
            classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
            positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
            probability_by_row = [float(row[positive_index]) for row in probabilities]
        version = file_version(self.models_dir / "best_model_for_yield.joblib")
        results = []
        for index, (value, raw_prediction) in enumerate(zip(values, raw_predictions, strict=True)):
            category = (
                "High"
                if raw_prediction in (1, True)
                else "Low"
                if raw_prediction in (0, False)
                else str(raw_prediction)
            )
            results.append(
                PredictionResult(
                    prediction_type=value.prediction_type,
                    electrocatalyst=value.electrocatalyst,
                    category=category,
                    probability=probability_by_row[index],
                    model_version=version,
                    input_data=value.model_dump(mode="json"),
                    warnings=warnings[index],
                )
            )
        return results, model_input

    def predict_many(self, values: list[PredictionInput]) -> list[PredictionResult]:
        return self.evaluate_many(values)[0]

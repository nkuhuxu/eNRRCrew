from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import load

from enrrcrew.predictors.common import build_feature_frames, file_version
from enrrcrew.schemas import PredictionInput, PredictionResult, PredictionType

CLUSTER_TO_FE = {3: "High", 0: "Low", 1: "Low", 2: "Low", 4: "Low", 5: "Low"}


@lru_cache(maxsize=4)
def _load_resources(models_dir_text: str) -> tuple[object, object, object, object, list[str]]:
    models_dir = Path(models_dir_text)
    return (
        load(models_dir / "scaler.joblib"),
        load(models_dir / "nearest_centroid.joblib"),
        load(models_dir / "str_to_comp.joblib"),
        load(models_dir / "element_property.joblib"),
        list(load(models_dir / "features.joblib")),
    )


class FEPredictor:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir.resolve()

    def predict(self, value: PredictionInput) -> PredictionResult:
        return self.predict_many([value])[0]

    def evaluate_many(
        self, values: list[PredictionInput]
    ) -> tuple[list[PredictionResult], np.ndarray, list[float]]:
        if any(value.prediction_type is not PredictionType.FE for value in values):
            raise ValueError("FEPredictor requires prediction_type='fe'")
        if not values:
            return [], np.empty((0, 0)), []
        scaler, centroid, converter, element_property, features = _load_resources(
            str(self.models_dir)
        )
        frame, warnings = build_feature_frames(
            values, features, converter, element_property
        )
        scaled = scaler.transform(frame)
        clusters = [int(cluster) for cluster in centroid.predict(scaled)]
        unknown = sorted(set(clusters).difference(CLUSTER_TO_FE))
        if unknown:
            raise ValueError(
                f"model returned unknown FE cluster {unknown[0]}; refusing to silently classify it"
            )
        if hasattr(centroid, "classes_") and hasattr(centroid, "centroids_"):
            classes = [int(value) for value in centroid.classes_]
            centers = np.asarray(centroid.centroids_, dtype=float)
            high_center = centers[classes.index(3)]
            low_centers = centers[
                [index for index, label in enumerate(classes) if label != 3]
            ]
            high_distance = np.linalg.norm(scaled - high_center, axis=1)
            low_distance = np.min(
                np.linalg.norm(
                    scaled[:, np.newaxis, :] - low_centers[np.newaxis, :, :], axis=2
                ),
                axis=1,
            )
            margins = (
                (low_distance - high_distance)
                / (low_distance + high_distance + 1e-12)
            ).clip(-1, 1)
        else:
            # Lightweight test doubles and legacy adapters may expose only
            # predict(). Preserve compatibility while keeping the real model's
            # distance-based margin for recommendation ranking.
            margins = np.asarray([1.0 if cluster == 3 else -1.0 for cluster in clusters])
        version = file_version(self.models_dir / "nearest_centroid.joblib")
        results = [
            PredictionResult(
                prediction_type=value.prediction_type,
                electrocatalyst=value.electrocatalyst,
                category=CLUSTER_TO_FE[cluster],
                cluster=cluster,
                model_version=version,
                input_data=value.model_dump(mode="json"),
                warnings=warnings[index],
            )
            for index, (value, cluster) in enumerate(zip(values, clusters, strict=True))
        ]
        return results, np.asarray(scaled), [float(margin) for margin in margins]

    def predict_many(self, values: list[PredictionInput]) -> list[PredictionResult]:
        return self.evaluate_many(values)[0]

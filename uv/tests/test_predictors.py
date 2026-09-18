from pathlib import Path

import numpy as np
import pandas as pd

from enrrcrew.predictors import fe as fe_module
from enrrcrew.predictors import yield_model as yield_module
from enrrcrew.predictors.fe import FEPredictor
from enrrcrew.predictors.yield_model import YieldPredictor
from enrrcrew.schemas import PredictionInput


class CompositionFeaturizer:
    def featurize_dataframe(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        result = dataframe.copy()
        result["composition"] = result[column]
        return result


class PropertyFeaturizer:
    def featurize_dataframe(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        return dataframe.copy()


class TransformOnlyScaler:
    def __init__(self) -> None:
        self.transform_calls = 0

    def transform(self, value: object) -> np.ndarray:
        self.transform_calls += 1
        return np.asarray(value, dtype=float)


class YieldModel:
    classes_ = np.array([0, 1])

    def predict(self, value: object) -> np.ndarray:
        return np.array([1])

    def predict_proba(self, value: object) -> np.ndarray:
        return np.array([[0.2, 0.8]])


class CentroidModel:
    def __init__(self, cluster: int = 3) -> None:
        self.cluster = cluster

    def predict(self, value: object) -> np.ndarray:
        return np.array([self.cluster])


def value(kind: str) -> PredictionInput:
    return PredictionInput.model_validate(
        {
            "prediction_type": kind,
            "applied_potential": -0.3,
            "electrocatalyst": "Fe-N-C",
            "elements": ["Fe", "N", "C"],
            "morphology": "porous nanosheet",
        }
    )


def test_yield_predictor_uses_saved_scaler_without_fitting(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "best_model_for_yield.joblib"
    model_path.write_bytes(b"model")
    scaler = TransformOnlyScaler()
    monkeypatch.setattr(
        yield_module,
        "_load_resources",
        lambda _: (
            scaler,
            YieldModel(),
            ["Applied Potential (NH3 Yield)"],
            ["Applied Potential (NH3 Yield)"],
            CompositionFeaturizer(),
            PropertyFeaturizer(),
        ),
    )
    result = YieldPredictor(tmp_path).predict(value("yield"))
    assert scaler.transform_calls == 1
    assert result.electrocatalyst == "Fe-N-C"
    assert result.category == "High"
    assert result.probability == 0.8


def test_fe_predictor_retains_catalyst_and_cluster(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "nearest_centroid.joblib"
    model_path.write_bytes(b"model")
    scaler = TransformOnlyScaler()
    monkeypatch.setattr(
        fe_module,
        "_load_resources",
        lambda _: (
            scaler,
            CentroidModel(),
            CompositionFeaturizer(),
            PropertyFeaturizer(),
            ["Applied Potential (Faraday Efficiency)"],
        ),
    )
    result = FEPredictor(tmp_path).predict(value("fe"))
    assert scaler.transform_calls == 1
    assert result.electrocatalyst == "Fe-N-C"
    assert result.cluster == 3
    assert result.category == "High"


def test_fe_predictor_rejects_unknown_cluster(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "nearest_centroid.joblib").write_bytes(b"model")
    monkeypatch.setattr(
        fe_module,
        "_load_resources",
        lambda _: (
            TransformOnlyScaler(),
            CentroidModel(99),
            CompositionFeaturizer(),
            PropertyFeaturizer(),
            ["Applied Potential (Faraday Efficiency)"],
        ),
    )
    try:
        FEPredictor(tmp_path).predict(value("fe"))
    except ValueError as exc:
        assert "unknown FE cluster" in str(exc)
    else:
        raise AssertionError("Unknown clusters must not be silently classified")


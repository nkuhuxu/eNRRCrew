from pathlib import Path

import pytest

from enrrcrew.predictors import FEPredictor, YieldPredictor
from enrrcrew.schemas import PredictionInput

ASSET_ROOT = Path(__file__).resolve().parents[2]


def prediction_input(kind: str) -> PredictionInput:
    return PredictionInput.model_validate(
        {
            "prediction_type": kind,
            "applied_potential": -0.3,
            "electrocatalyst": "Fe-N-C",
            "elements": ["Fe", "N", "C"],
            "morphology": "porous nanosheet",
            "ph_categories": ["neutral"],
            "electrolytes": ["pbs"],
            "n15_labeling": True,
        }
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kind", "predictor_type"),
    [("yield", YieldPredictor), ("fe", FEPredictor)],
)
def test_persisted_models_predict(kind: str, predictor_type: type) -> None:
    models = ASSET_ROOT / "models"
    if not models.is_dir():
        pytest.skip("legacy model assets are unavailable")
    result = predictor_type(models).predict(prediction_input(kind))
    assert result.electrocatalyst == "Fe-N-C"
    assert result.category in {"High", "Low"}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kind", "predictor_type"),
    [("yield", YieldPredictor), ("fe", FEPredictor)],
)
def test_persisted_models_batch_matches_single(kind: str, predictor_type: type) -> None:
    models = ASSET_ROOT / "models"
    if not models.is_dir():
        pytest.skip("legacy model assets are unavailable")
    predictor = predictor_type(models)
    value = prediction_input(kind)
    single = predictor.predict(value)
    batch = predictor.predict_many([value, value])
    assert [item.category for item in batch] == [single.category, single.category]


def test_main_environment_does_not_depend_on_graphrag_runtime() -> None:
    project = Path(__file__).resolve().parents[1]
    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert '"graphrag==' not in pyproject
    assert '"lancedb==' not in pyproject
    assert (project / "graphrag_service" / "uv.lock").is_file()

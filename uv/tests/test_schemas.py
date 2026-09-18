import pytest
from pydantic import ValidationError

from enrrcrew.schemas import PredictionInput


def test_prediction_input_normalizes_elements_and_categories() -> None:
    value = PredictionInput.model_validate(
        {
            "prediction_type": "yield",
            "applied_potential": -0.3,
            "electrocatalyst": " CoMo/NC ",
            "elements": ["co", "MO", "C", "N", "Co"],
            "morphology": "porous nanofibers",
            "ph_categories": ["acidic"],
            "electrolytes": ["pbs"],
        }
    )
    assert value.elements == ["Co", "Mo", "C", "N"]
    assert value.electrocatalyst == "CoMo/NC"


@pytest.mark.parametrize("symbol", ["Cobalt", "Xx", ""]) 
def test_prediction_input_rejects_invalid_element_symbols(symbol: str) -> None:
    with pytest.raises(ValidationError):
        PredictionInput.model_validate(
            {
                "prediction_type": "fe",
                "applied_potential": -0.3,
                "electrocatalyst": "test",
                "elements": [symbol],
            }
        )


def test_prediction_input_rejects_invalid_potential() -> None:
    with pytest.raises(ValidationError):
        PredictionInput.model_validate(
            {
                "prediction_type": "yield",
                "applied_potential": -30,
                "electrocatalyst": "test",
                "elements": ["Fe"],
            }
        )


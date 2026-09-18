import pandas as pd

from enrrcrew.predictors.common import (
    build_feature_frame,
    categorize_structure,
    input_to_frame,
)
from enrrcrew.schemas import PredictionInput


class CompositionFeaturizer:
    def featurize_dataframe(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        result = dataframe.copy()
        result["composition"] = result[column]
        return result


class PropertyFeaturizer:
    def featurize_dataframe(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        result = dataframe.copy()
        result["MagpieData mean Number"] = 27.0
        return result


def make_input(morphology: str = "porous nanofibers") -> PredictionInput:
    return PredictionInput.model_validate(
        {
            "prediction_type": "yield",
            "applied_potential": -0.25,
            "electrocatalyst": "CoMo/NC",
            "elements": ["Co", "Mo", "C", "N"],
            "morphology": morphology,
            "ph_categories": ["acidic"],
            "electrolytes": ["pbs"],
            "n15_labeling": True,
        }
    )


def test_input_mapping_preserves_ph_and_electrolyte_flags() -> None:
    frame = input_to_frame(make_input())
    assert frame.loc[0, "pH_acidic"] == 1
    assert frame.loc[0, "pH_alkaline"] == 0
    assert frame.loc[0, "Electrolyte without concentration_pbs"] == 1
    assert frame.loc[0, "N-15 labeling_mentioned"] == 1


def test_feature_frame_aligns_training_order() -> None:
    feature_names = [
        "pH_acidic",
        "Applied Potential (NH3 Yield)",
        "Structure Type_Porous Structures",
        "MagpieData mean Number",
        "missing training feature",
    ]
    frame, warnings = build_feature_frame(
        make_input("porous catalyst"),
        feature_names,
        CompositionFeaturizer(),
        PropertyFeaturizer(),
    )
    assert frame.columns.tolist() == feature_names
    assert frame.iloc[0].tolist() == [1.0, -0.25, 1.0, 27.0, 0.0]
    assert any("filled with zero" in warning for warning in warnings)


def test_empty_morphology_is_explicitly_reported() -> None:
    frame, warnings = build_feature_frame(
        make_input(""), ["Structure Type_Other"], CompositionFeaturizer(), PropertyFeaturizer()
    )
    assert frame.iloc[0, 0] == 0
    assert any("Morphology was empty" in warning for warning in warnings)


def test_structure_categories_cover_supported_morphologies() -> None:
    assert categorize_structure("2D MoS2 nanosheet") == "Nanosheets"
    assert categorize_structure("hollow porous sphere") == "Hollow Structures"
    assert categorize_structure("unclassified bulk material") == "Other"

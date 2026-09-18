from __future__ import annotations

from typing import Any

import streamlit as st
from pydantic import ValidationError

from enrrcrew.schemas import ELECTROLYTES, PH_CATEGORIES, PredictionInput, PredictionType


def _prefix(prediction_type: PredictionType) -> str:
    return f"prediction_{prediction_type.value}"


def apply_prediction_defaults(value: PredictionInput) -> None:
    prefix = _prefix(value.prediction_type)
    defaults = {
        f"{prefix}_potential": value.applied_potential,
        f"{prefix}_catalyst": value.electrocatalyst,
        f"{prefix}_elements": ", ".join(value.elements),
        f"{prefix}_morphology": value.morphology,
        f"{prefix}_ph": sorted(value.ph_categories),
        f"{prefix}_electrolytes": sorted(value.electrolytes),
        f"{prefix}_n15": value.n15_labeling,
    }
    for key, item in defaults.items():
        st.session_state[key] = item


def render_prediction_form(
    prediction_type: PredictionType,
) -> tuple[bool, PredictionInput | None]:
    prefix = _prefix(prediction_type)
    label = "NH₃ yield" if prediction_type is PredictionType.YIELD else "Faradaic efficiency"
    with st.form(f"{prefix}_form", border=True):
        st.markdown(f"### {label} input")
        left, right = st.columns(2)
        with left:
            potential = st.number_input(
                "Applied potential (V)",
                min_value=-10.0,
                max_value=10.0,
                value=-0.3,
                step=0.05,
                key=f"{prefix}_potential",
            )
            catalyst = st.text_input("Electrocatalyst", key=f"{prefix}_catalyst")
            elements = st.text_input(
                "Elements (comma-separated symbols)",
                placeholder="Co, Mo, C, N",
                key=f"{prefix}_elements",
            )
            morphology = st.text_input(
                "Morphology",
                placeholder="porous nanofibers",
                key=f"{prefix}_morphology",
            )
        with right:
            ph = st.multiselect("pH categories", PH_CATEGORIES, key=f"{prefix}_ph")
            electrolytes = st.multiselect(
                "Electrolytes", ELECTROLYTES, key=f"{prefix}_electrolytes"
            )
            n15 = st.checkbox("N-15 labeling mentioned", key=f"{prefix}_n15")
        submitted = st.form_submit_button("Validate & predict", use_container_width=True)
    if not submitted:
        return False, None
    payload: dict[str, Any] = {
        "prediction_type": prediction_type,
        "applied_potential": potential,
        "electrocatalyst": catalyst,
        "elements": [item.strip() for item in elements.split(",") if item.strip()],
        "morphology": morphology,
        "ph_categories": set(ph),
        "electrolytes": set(electrolytes),
        "n15_labeling": n15,
    }
    try:
        return True, PredictionInput.model_validate(payload)
    except ValidationError as exc:
        st.error("Please correct the highlighted scientific input.")
        for error in exc.errors():
            location = ".".join(str(item) for item in error["loc"])
            st.caption(f"{location}: {error['msg']}")
        return True, None


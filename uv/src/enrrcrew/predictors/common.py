from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from enrrcrew.schemas import ELECTROLYTES, PH_CATEGORIES, PredictionInput, PredictionType


class Featurizer(Protocol):
    def featurize_dataframe(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame: ...


def _featurize(
    featurizer: Featurizer, dataframe: pd.DataFrame, column: str
) -> pd.DataFrame:
    method = featurizer.featurize_dataframe
    parameters = inspect.signature(method).parameters.values()
    supports_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    supports_pbar = "pbar" in inspect.signature(method).parameters or supports_kwargs
    if supports_pbar:
        return method(dataframe, column, pbar=False)
    return method(dataframe, column)


STRUCTURE_TYPES = (
    "Hollow Structures",
    "Hybrids and Composites",
    "Nanocubes",
    "Nanofibers",
    "Nanoparticles",
    "Nanorods",
    "Nanosheets",
    "Nanotubes",
    "Nanowalls",
    "Nanowires",
    "Other",
    "Porous Structures",
)


def categorize_structure(morphology: str) -> str | None:
    value = morphology.strip().lower()
    if not value:
        return None
    checks = (
        (("nanoparticle", "quantum dots", "nanodots"), "Nanoparticles"),
        (("nanorod",), "Nanorods"),
        (("nanowire",), "Nanowires"),
        (("nanosheet", "2d"), "Nanosheets"),
        (("nanofiber",), "Nanofibers"),
        (("nanocube",), "Nanocubes"),
        (("hollow",), "Hollow Structures"),
        (("porous",), "Porous Structures"),
        (("hybrid", "composite"), "Hybrids and Composites"),
        (("nanotube",), "Nanotubes"),
        (("nanowall",), "Nanowalls"),
    )
    for needles, category in checks:
        if any(needle in value for needle in needles):
            return category
    return "Other"


def input_to_frame(value: PredictionInput) -> pd.DataFrame:
    potential_column = (
        "Applied Potential (NH3 Yield)"
        if value.prediction_type is PredictionType.YIELD
        else "Applied Potential (Faraday Efficiency)"
    )
    row: dict[str, object] = {
        potential_column: value.applied_potential,
        "Electrocatalyst": value.electrocatalyst,
        "Morphology of electrocatalyst": value.morphology,
        "N-15 labeling_mentioned": int(value.n15_labeling),
    }
    for index in range(7):
        row[f"Elements of electrocatalyst_{index}"] = (
            value.elements[index] if index < len(value.elements) else None
        )
    for category in PH_CATEGORIES:
        row[f"pH_{category}"] = int(category in value.ph_categories)
    for electrolyte in ELECTROLYTES:
        row[f"Electrolyte without concentration_{electrolyte}"] = int(
            electrolyte in value.electrolytes
        )
    return pd.DataFrame([row])


def inputs_to_frame(values: Sequence[PredictionInput]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()
    return pd.concat([input_to_frame(value) for value in values], ignore_index=True)


def build_feature_frame(
    value: PredictionInput,
    feature_names: Sequence[str],
    str_to_composition: Featurizer,
    element_property: Featurizer,
) -> tuple[pd.DataFrame, list[str]]:
    frames, warnings = build_feature_frames(
        [value], feature_names, str_to_composition, element_property
    )
    return frames, warnings[0]


def build_feature_frames(
    values: Sequence[PredictionInput],
    feature_names: Sequence[str],
    str_to_composition: Featurizer,
    element_property: Featurizer,
) -> tuple[pd.DataFrame, list[list[str]]]:
    if not values:
        return pd.DataFrame(columns=list(feature_names), dtype="float64"), []
    frame = inputs_to_frame(values)
    warnings: list[list[str]] = [[] for _ in values]
    # Matminer defaults to multiprocessing. A single prediction is faster and
    # substantially safer under Streamlit on Windows when kept in-process.
    for featurizer in (str_to_composition, element_property):
        set_n_jobs = getattr(featurizer, "set_n_jobs", None)
        if callable(set_n_jobs):
            set_n_jobs(1)
    element_columns = [f"Elements of electrocatalyst_{index}" for index in range(7)]
    frame["formula"] = frame[element_columns].apply(
        lambda row: "".join(item for item in row if isinstance(item, str)), axis=1
    )
    frame = _featurize(str_to_composition, frame, "formula")
    frame = _featurize(element_property, frame, "composition")

    structures = [categorize_structure(value.morphology) for value in values]
    for category in STRUCTURE_TYPES:
        frame[f"Structure Type_{category}"] = [int(structure == category) for structure in structures]
    for index, structure in enumerate(structures):
        if structure is None:
            warnings[index].append(
                "Morphology was empty; all structure-type features were set to zero."
            )

    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        for item in warnings:
            item.append(f"{len(missing)} absent training features were filled with zero.")
    aligned = frame.reindex(columns=list(feature_names), fill_value=0)
    aligned = aligned.apply(pd.to_numeric, errors="coerce")
    if aligned.isna().any(axis=None):
        bad_columns = aligned.columns[aligned.isna().any()].tolist()
        raise ValueError(f"non-numeric or missing feature values: {bad_columns[:8]}")
    return aligned.astype("float64"), warnings


def file_version(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.name}:{digest}"

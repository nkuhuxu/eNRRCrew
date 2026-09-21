from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionType(StrEnum):
    YIELD = "yield"
    FE = "fe"


PH_CATEGORIES = (
    "acidic",
    "alkaline",
    "ionic liquid",
    "khco3",
    "li tfsi",
    "nabf4",
    "neutral",
    "weak acid",
)

ELECTROLYTES = (
    "h2so4",
    "hcl",
    "k2so4",
    "kclo4",
    "koh",
    "li2so4",
    "licl",
    "liclo4",
    "lioh",
    "na2so4",
    "naoh",
    "pbs",
)

ELEMENT_SYMBOLS = frozenset(
    ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
)


class PredictionInput(BaseModel):
    prediction_type: PredictionType
    applied_potential: float = Field(ge=-10, le=10)
    electrocatalyst: str = Field(min_length=1, max_length=240)
    elements: list[str] = Field(min_length=1, max_length=7)
    morphology: str = Field(default="", max_length=500)
    ph_categories: set[str] = Field(default_factory=set)
    electrolytes: set[str] = Field(default_factory=set)
    n15_labeling: bool = False

    @field_validator("electrocatalyst")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("electrocatalyst cannot be blank")
        return value

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            symbol = value.strip().capitalize()
            if symbol not in ELEMENT_SYMBOLS:
                raise ValueError(f"unknown element symbol: {value}")
            if symbol not in normalized:
                normalized.append(symbol)
        if not normalized:
            raise ValueError("at least one valid element is required")
        return normalized

    @model_validator(mode="after")
    def validate_categories(self) -> PredictionInput:
        bad_ph = self.ph_categories.difference(PH_CATEGORIES)
        bad_electrolytes = self.electrolytes.difference(ELECTROLYTES)
        if bad_ph:
            raise ValueError(f"unsupported pH categories: {sorted(bad_ph)}")
        if bad_electrolytes:
            raise ValueError(f"unsupported electrolytes: {sorted(bad_electrolytes)}")
        return self


class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction_type: PredictionType
    electrocatalyst: str
    category: str
    probability: float | None = Field(default=None, ge=0, le=1)
    cluster: int | None = None
    model_version: str
    input_data: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class RecommendationMode(StrEnum):
    KNOWN = "known"
    EXPLORE = "explore"
    HYBRID = "hybrid"


class CandidateOrigin(StrEnum):
    KNOWN = "known"
    GENERATED = "generated"


class RecommendationRequest(BaseModel):
    mode: RecommendationMode = RecommendationMode.HYBRID
    allowed_elements: set[str] = Field(min_length=1)
    forbidden_elements: set[str] = Field(default_factory=set)
    max_elements: int = Field(default=4, ge=1, le=7)
    morphologies: set[str] = Field(default_factory=set)
    ph_categories: set[str] = Field(default_factory=set)
    electrolytes: set[str] = Field(default_factory=set)
    potential_min: float = Field(default=-0.8, ge=-10, le=10)
    potential_max: float = Field(default=0.0, ge=-10, le=10)
    potential_step: float = Field(default=0.1, ge=0.05, le=0.5)
    generated_candidate_limit: int = Field(default=500, ge=0, le=2000)
    result_limit: int = Field(default=20, ge=1, le=100)

    @field_validator("allowed_elements", "forbidden_elements")
    @classmethod
    def validate_element_sets(cls, values: set[str]) -> set[str]:
        normalized = {value.strip().capitalize() for value in values if value.strip()}
        invalid = normalized.difference(ELEMENT_SYMBOLS)
        if invalid:
            raise ValueError(f"unknown element symbols: {sorted(invalid)}")
        return normalized

    @model_validator(mode="after")
    def validate_recommendation(self) -> RecommendationRequest:
        if not self.allowed_elements:
            raise ValueError("at least one allowed element is required")
        overlap = self.allowed_elements.intersection(self.forbidden_elements)
        if overlap:
            raise ValueError(f"allowed and forbidden elements overlap: {sorted(overlap)}")
        if self.potential_min > self.potential_max:
            raise ValueError("potential_min cannot be greater than potential_max")
        grid_size = (
            math.ceil(
                (self.potential_max - self.potential_min) / self.potential_step - 1e-12
            )
            + 1
        )
        if grid_size > 25:
            raise ValueError("the potential grid may contain at most 25 points")
        bad_ph = self.ph_categories.difference(PH_CATEGORIES)
        bad_electrolytes = self.electrolytes.difference(ELECTROLYTES)
        if bad_ph:
            raise ValueError(f"unsupported pH categories: {sorted(bad_ph)}")
        if bad_electrolytes:
            raise ValueError(f"unsupported electrolytes: {sorted(bad_electrolytes)}")
        return self

    def potential_grid(self) -> list[float]:
        count = math.floor(
            (self.potential_max - self.potential_min) / self.potential_step + 1e-12
        )
        values = [self.potential_min + index * self.potential_step for index in range(count + 1)]
        if not values or values[-1] < self.potential_max - 1e-9:
            values.append(self.potential_max)
        return [round(value, 6) for value in values if value <= self.potential_max + 1e-9]


class RecommendationCandidate(BaseModel):
    candidate_id: str
    display_name: str
    origin: CandidateOrigin
    elements: list[str]
    morphology: str = ""
    applied_potential: float
    ph_categories: set[str] = Field(default_factory=set)
    electrolytes: set[str] = Field(default_factory=set)
    source_rows: list[int] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    candidate: RecommendationCandidate
    yield_result: PredictionResult
    fe_result: PredictionResult
    yield_high_score: float | None = Field(default=None, ge=0, le=1)
    fe_centroid_margin: float = Field(ge=-1, le=1)
    combined_signal: float = Field(ge=0, le=1)
    applicability_score: float = Field(ge=0, le=1)
    applicability_status: Literal["in_domain", "borderline", "out_of_domain"]
    novelty_score: float = Field(ge=0, le=1)
    nearest_known_candidates: list[dict[str, Any]] = Field(default_factory=list)
    evidence_status: Literal["not_requested", "available", "unavailable"] = "not_requested"
    warnings: list[str] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request: RecommendationRequest
    recommendations: list[RecommendationItem]
    near_misses: list[RecommendationItem]
    rejected_counts: dict[str, int]
    model_versions: dict[str, str]
    run_id: str


class SandboxResult(BaseModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    output_files: list[Path] = Field(default_factory=list)
    timed_out: bool = False

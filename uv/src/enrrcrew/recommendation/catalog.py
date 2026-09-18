from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import pandas as pd

from enrrcrew.predictors.common import categorize_structure
from enrrcrew.schemas import (
    ELECTROLYTES,
    ELEMENT_SYMBOLS,
    PH_CATEGORIES,
    CandidateOrigin,
    RecommendationCandidate,
)

ELEMENT_COLUMNS = tuple(f"Elements of electrocatalyst_{index}" for index in range(7))


def candidate_signature(candidate: RecommendationCandidate) -> str:
    parts = (
        ",".join(sorted(candidate.elements)),
        categorize_structure(candidate.morphology) or "",
        ",".join(sorted(candidate.ph_categories)),
        ",".join(sorted(candidate.electrolytes)),
        f"{candidate.applied_potential:.6f}",
    )
    return "|".join(parts)


def candidate_id(signature: str) -> str:
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class CatalystCatalog:
    candidates: tuple[RecommendationCandidate, ...]
    element_counts: Counter[str]
    cooccurrence: dict[tuple[str, str], int]

    @property
    def signatures(self) -> frozenset[str]:
        return frozenset(candidate_signature(candidate) for candidate in self.candidates)

    def default_allowed_elements(self, minimum_count: int = 10) -> list[str]:
        return sorted(
            element for element, count in self.element_counts.items() if count >= minimum_count
        )


def _active_categories(row: pd.Series, prefix: str, choices: tuple[str, ...]) -> set[str]:
    active = set()
    for choice in choices:
        value = pd.to_numeric(row.get(f"{prefix}{choice}"), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            active.add(choice)
    return active


@lru_cache(maxsize=4)
def load_catalog(csv_path_text: str) -> CatalystCatalog:
    csv_path = Path(csv_path_text)
    frame = pd.read_csv(csv_path)
    merged: dict[str, RecommendationCandidate] = {}
    element_counts: Counter[str] = Counter()
    cooccurrence: Counter[tuple[str, str]] = Counter()

    for row_index, row in frame.iterrows():
        elements = []
        for column in ELEMENT_COLUMNS:
            raw = row.get(column)
            if not isinstance(raw, str):
                continue
            symbol = raw.strip().capitalize()
            if symbol in ELEMENT_SYMBOLS and symbol not in elements:
                elements.append(symbol)
        if not elements:
            continue
        element_counts.update(elements)
        for left, right in combinations(sorted(elements), 2):
            cooccurrence[(left, right)] += 1

        morphology_raw = row.get("Morphology of electrocatalyst")
        morphology = categorize_structure(morphology_raw if isinstance(morphology_raw, str) else "")
        ph_categories = _active_categories(row, "pH_", PH_CATEGORIES)
        electrolytes = _active_categories(
            row, "Electrolyte without concentration_", ELECTROLYTES
        )
        name_raw = row.get("Electrocatalyst")
        display_name = str(name_raw).strip() if pd.notna(name_raw) else "Unknown catalyst"
        potentials = set()
        for column in (
            "Applied Potential (Faraday Efficiency)",
            "Applied Potential (NH3 Yield)",
        ):
            potential = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(potential) and -10 <= float(potential) <= 10:
                potentials.add(round(float(potential), 6))

        for potential in sorted(potentials):
            provisional = RecommendationCandidate(
                candidate_id="pending",
                display_name=display_name or "Unknown catalyst",
                origin=CandidateOrigin.KNOWN,
                elements=elements,
                morphology=morphology or "",
                applied_potential=potential,
                ph_categories=ph_categories,
                electrolytes=electrolytes,
                source_rows=[int(row_index)],
            )
            signature = candidate_signature(provisional)
            existing = merged.get(signature)
            if existing is None:
                merged[signature] = provisional.model_copy(
                    update={"candidate_id": candidate_id(signature)}
                )
            else:
                rows = sorted(set(existing.source_rows).union({int(row_index)}))
                merged[signature] = existing.model_copy(update={"source_rows": rows})

    return CatalystCatalog(
        candidates=tuple(merged[key] for key in sorted(merged)),
        element_counts=element_counts,
        cooccurrence=dict(cooccurrence),
    )

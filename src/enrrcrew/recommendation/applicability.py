from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(slots=True)
class DomainSpace:
    reference: np.ndarray
    neighbor_distances: np.ndarray
    p95: float
    p99: float
    neighbors: NearestNeighbors

    @classmethod
    def from_reference(cls, reference: np.ndarray) -> DomainSpace:
        matrix = np.asarray(reference, dtype=float)
        if matrix.ndim != 2 or len(matrix) == 0:
            raise ValueError("applicability reference matrix cannot be empty")
        count = min(6, len(matrix))
        neighbors = NearestNeighbors(n_neighbors=count).fit(matrix)
        distances = neighbors.kneighbors(matrix, return_distance=True)[0]
        leave_one_out = []
        for row in distances:
            positive = row[row > 1e-12]
            leave_one_out.append(float(positive[0] if len(positive) else row[-1]))
        distribution = np.sort(np.asarray(leave_one_out, dtype=float))
        return cls(
            reference=matrix,
            neighbor_distances=distribution,
            p95=max(float(np.quantile(distribution, 0.95)), 1e-12),
            p99=max(float(np.quantile(distribution, 0.99)), 1e-12),
            neighbors=neighbors,
        )

    def assess(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(vectors, dtype=float)
        distances = self.neighbors.kneighbors(matrix, n_neighbors=1, return_distance=True)[0][
            :, 0
        ]
        percentiles = np.searchsorted(
            self.neighbor_distances, distances, side="right"
        ) / len(self.neighbor_distances)
        return distances, percentiles


@dataclass(slots=True)
class ApplicabilityDomain:
    yield_space: DomainSpace
    fe_space: DomainSpace

    @classmethod
    def from_reference(
        cls, yield_reference: np.ndarray, fe_reference: np.ndarray
    ) -> ApplicabilityDomain:
        return cls(
            yield_space=DomainSpace.from_reference(yield_reference),
            fe_space=DomainSpace.from_reference(fe_reference),
        )

    def assess(
        self, yield_vectors: np.ndarray, fe_vectors: np.ndarray
    ) -> list[tuple[float, str]]:
        yield_distances, yield_percentiles = self.yield_space.assess(yield_vectors)
        fe_distances, fe_percentiles = self.fe_space.assess(fe_vectors)
        assessments = []
        for yd, fd, yp, fp in zip(
            yield_distances,
            fe_distances,
            yield_percentiles,
            fe_percentiles,
            strict=True,
        ):
            out = yd > self.yield_space.p99 or fd > self.fe_space.p99
            borderline = yd > self.yield_space.p95 or fd > self.fe_space.p95
            status = "out_of_domain" if out else "borderline" if borderline else "in_domain"
            score = max(0.0, 1.0 - float(max(yp, fp)))
            assessments.append((score, status))
        return assessments

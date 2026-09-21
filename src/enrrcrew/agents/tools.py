from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.predictors import FEPredictor, YieldPredictor
from enrrcrew.recommendation import RecommendationService
from enrrcrew.schemas import PredictionType
from enrrcrew.services import (
    MissingPredictionFields,
    PredictionTextExtractor,
    RagService,
    RecommendationTextExtractor,
)

from .csv_analyst import CsvAnalysisAgent
from .models import AgentArtifacts, AgentRequest, AgentTraceEvent

MAX_TOOL_TEXT = 12_000


@dataclass(frozen=True, slots=True)
class AgentToolContext:
    settings: AppSettings
    workspace: SessionWorkspace
    rag: RagService
    yield_predictor: YieldPredictor
    fe_predictor: FEPredictor
    recommendation_service: RecommendationService
    prediction_extractor: PredictionTextExtractor
    recommendation_extractor: RecommendationTextExtractor
    csv_agent: CsvAnalysisAgent
    datasets: dict[str, Path]


def _trim(value: str, limit: int = MAX_TOOL_TEXT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[tool output truncated]"


class AgentToolbox:
    def __init__(self, request: AgentRequest, context: AgentToolContext):
        self.request = request
        self.context = context
        self.artifacts = AgentArtifacts()
        self.trace: list[AgentTraceEvent] = []
        self.outcomes: list[str] = []
        self.results: dict[str, dict[str, object]] = {}

    def _safe_text(self, value: str, limit: int = MAX_TOOL_TEXT) -> str:
        safe = value
        replacements = (
            self.context.settings.api_key,
            str(self.context.settings.asset_root),
            str(self.context.settings.runtime_root),
            str(self.context.workspace.root),
        )
        for private_value in replacements:
            if private_value:
                safe = safe.replace(private_value, "[REDACTED]")
                safe = safe.replace(private_value.replace("\\", "/"), "[REDACTED]")
        return _trim(safe, limit)

    def _run(self, agent: str, tool: str, operation: Any) -> str:
        started = time.monotonic()
        self.trace.append(
            AgentTraceEvent(
                round_index=0,
                agent=agent,
                event_type="tool_call",
                tool=tool,
                status="started",
                summary=f"{tool} started",
            )
        )
        try:
            payload = operation()
        except MissingPredictionFields as exc:
            payload = {
                "status": "needs_input",
                "missing_fields": exc.fields,
                "message": str(exc),
            }
        except FileNotFoundError:
            payload = {
                "status": "failed",
                "message": "A required session or application resource was not found",
            }
        except UnicodeError:
            payload = {
                "status": "failed",
                "message": f"{tool} could not encode or decode scientific text",
            }
        except ValueError as exc:
            payload = {"status": "needs_input", "message": self._safe_text(str(exc))}
        except Exception as exc:
            payload = {
                "status": "failed",
                "message": f"{tool} failed ({type(exc).__name__})",
            }
        status = str(payload.get("status", "completed"))
        self.outcomes.append(status)
        self.results[tool] = payload
        trace_status = "completed" if status in {"completed", "needs_input"} else "failed"
        elapsed = int((time.monotonic() - started) * 1000)
        self.trace.append(
            AgentTraceEvent(
                round_index=0,
                agent=agent,
                event_type="tool_result" if trace_status == "completed" else "error",
                tool=tool,
                status=trace_status,
                summary=self._safe_text(str(payload.get("message", status)), 500),
                elapsed_ms=elapsed,
            )
        )
        return self._safe_text(
            json.dumps(payload, ensure_ascii=False, default=str), MAX_TOOL_TEXT
        )

    def search_evidence(
        self,
        question: str,
        mode: str = "",
        community: int = -1,
        response_type: str = "",
    ) -> str:
        def operation() -> dict[str, object]:
            selected_mode = mode if mode in {"local", "global"} else self.request.rag_mode
            selected_community = community if community >= 0 else self.request.community
            selected_response = response_type.strip() or self.request.response_type
            if hasattr(self.context.rag, "search_result"):
                result = self.context.rag.search_result(
                    question,
                    selected_mode,
                    selected_community,
                    selected_response,
                )
                answer = result.answer
                citations = [item.model_dump(mode="json") for item in result.citations]
                index_version = result.index_version
            else:
                answer = self.context.rag.search(
                    question,
                    selected_mode,
                    selected_community,
                    selected_response,
                )
                citations = []
                index_version = "unknown"
            return {
                "status": "completed",
                "message": "GraphRAG evidence retrieved",
                "evidence": _trim(answer),
                "index_version": index_version,
                "citations": citations,
            }

        return self._run("Retriever", "search_evidence", operation)

    def _predict(self, description: str, prediction_type: PredictionType) -> str:
        agent = "Yield_Predictor" if prediction_type is PredictionType.YIELD else "FE_Predictor"
        tool = "predict_yield" if prediction_type is PredictionType.YIELD else "predict_fe"

        def operation() -> dict[str, object]:
            value = self.context.prediction_extractor.extract(description, prediction_type)
            predictor = (
                self.context.yield_predictor
                if prediction_type is PredictionType.YIELD
                else self.context.fe_predictor
            )
            result = predictor.predict(value)
            if prediction_type is PredictionType.YIELD:
                self.artifacts.yield_result = result
            else:
                self.artifacts.fe_result = result
            return {
                "status": "completed",
                "message": f"{prediction_type.value} prediction completed",
                "result": result.model_dump(mode="json"),
            }

        return self._run(agent, tool, operation)

    def predict_yield(self, description: str) -> str:
        return self._predict(description, PredictionType.YIELD)

    def predict_fe(self, description: str) -> str:
        return self._predict(description, PredictionType.FE)

    def recommend_catalysts(self, description: str) -> str:
        def operation() -> dict[str, object]:
            defaults = self.context.recommendation_service.default_allowed_elements()
            request = self.context.recommendation_extractor.extract(description, defaults)
            result = self.context.recommendation_service.recommend(request)
            self.artifacts.recommendation_result = result
            top_items = result.recommendations[:5]
            return {
                "status": "completed",
                "message": "catalyst recommendation completed",
                "run_id": result.run_id,
                "assumptions": request.model_dump(mode="json"),
                "recommendations": [
                    {
                        "name": item.candidate.display_name,
                        "origin": item.candidate.origin.value,
                        "potential": item.candidate.applied_potential,
                        "yield": item.yield_result.category,
                        "fe": item.fe_result.category,
                        "applicability": item.applicability_status,
                        "combined_signal": item.combined_signal,
                    }
                    for item in top_items
                ],
                "near_miss_count": len(result.near_misses),
            }

        return self._run(
            "Recommendation_Specialist", "recommend_catalysts", operation
        )

    def draft_csv_analysis(self, question: str, dataset_alias: str = "curated") -> str:
        def operation() -> dict[str, object]:
            alias = dataset_alias if dataset_alias in self.context.datasets else ""
            if not alias:
                raise ValueError(
                    "dataset_alias must be 'curated' or an uploaded session dataset"
                )
            path = self.context.datasets[alias]
            code = self.context.csv_agent.generate_code(question, path, "dataset.csv")
            self.artifacts.csv_code = code
            return {
                "status": "completed",
                "message": "CSV analysis code drafted; review it before Docker execution",
                "dataset": alias,
                "code": _trim(code),
            }

        return self._run("CSV_Analyst", "draft_csv_analysis", operation)

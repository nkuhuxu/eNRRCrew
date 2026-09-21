from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from enrrcrew.schemas import PredictionResult, RecommendationResult

from .router import ConversationRoute, RouteStrategy


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    rag_mode: Literal["local", "global"] = "local"
    community: int = Field(default=0, ge=0)
    response_type: str = Field(default="single paragraph", min_length=1, max_length=100)
    active_dataset: Literal["curated", "session_upload"] = "curated"


class AgentTraceEvent(BaseModel):
    round_index: int = Field(ge=0)
    agent: str
    event_type: Literal["message", "tool_call", "tool_result", "error"]
    tool: str | None = None
    status: Literal["started", "completed", "rejected", "failed"]
    summary: str
    elapsed_ms: int | None = Field(default=None, ge=0)


class AgentArtifacts(BaseModel):
    yield_result: PredictionResult | None = None
    fe_result: PredictionResult | None = None
    recommendation_result: RecommendationResult | None = None
    csv_code: str | None = None


class AgentRunResult(BaseModel):
    status: Literal["completed", "needs_input", "partial", "failed"]
    answer: str
    strategy: RouteStrategy
    detected_routes: list[ConversationRoute] = Field(default_factory=list)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    artifacts: AgentArtifacts = Field(default_factory=AgentArtifacts)
    rounds: int = Field(default=0, ge=0)

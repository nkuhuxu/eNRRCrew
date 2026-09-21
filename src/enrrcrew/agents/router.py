from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field


class ConversationRoute(StrEnum):
    RETRIEVAL = "retrieval"
    CSV = "csv"
    YIELD_PREDICTION = "yield_prediction"
    FE_PREDICTION = "fe_prediction"
    CATALYST_RECOMMENDATION = "catalyst_recommendation"


class RouteStrategy(StrEnum):
    FAST_PATH = "fast_path"
    GROUP_CHAT = "group_chat"


class RouteDecision(BaseModel):
    strategy: RouteStrategy
    detected_routes: list[ConversationRoute] = Field(default_factory=list)
    reason: str


_RECOMMENDATION_TERMS = (
    "recommend",
    "recommendation",
    "suggest catalyst",
    "推荐",
    "候选催化剂",
)
_YIELD_TERMS = ("yield", "nh3 yield", "ammonia yield", "产率", "产量", "氨产率")
_FE_TERMS = (
    "faradaic efficiency",
    "faraday efficiency",
    "法拉第效率",
    "选择性",
)
_PREDICTION_TERMS = ("predict", "prediction", "预测", "分类")
_CSV_TERMS = ("csv", "dataset", "data analysis", "数据集", "数据分析", "表格分析")
_RETRIEVAL_TERMS = ("retrieve", "search", "literature", "检索", "搜索", "文献", "查找")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if not term.isascii() and term in text:
            return True
        if term.isascii() and re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text
        ):
            return True
    return False


def _mentions_fe(text: str) -> bool:
    padded = f" {text} "
    return _contains_any(text, _FE_TERMS) or " fe " in padded


def decide_route(message: str) -> RouteDecision:
    text = " ".join(message.casefold().split())
    detected: list[ConversationRoute] = []

    if _contains_any(text, _RECOMMENDATION_TERMS):
        detected.append(ConversationRoute.CATALYST_RECOMMENDATION)
    if _contains_any(text, _PREDICTION_TERMS) and _contains_any(text, _YIELD_TERMS):
        detected.append(ConversationRoute.YIELD_PREDICTION)
    if _contains_any(text, _PREDICTION_TERMS) and _mentions_fe(text):
        detected.append(ConversationRoute.FE_PREDICTION)
    if _contains_any(text, _CSV_TERMS):
        detected.append(ConversationRoute.CSV)
    if _contains_any(text, _RETRIEVAL_TERMS):
        detected.append(ConversationRoute.RETRIEVAL)

    unique = list(dict.fromkeys(detected))
    if len(unique) == 1:
        return RouteDecision(
            strategy=RouteStrategy.FAST_PATH,
            detected_routes=unique,
            reason="one explicit intent was detected",
        )
    if len(unique) > 1:
        return RouteDecision(
            strategy=RouteStrategy.GROUP_CHAT,
            detected_routes=unique,
            reason="multiple intents require coordinated execution",
        )
    return RouteDecision(
        strategy=RouteStrategy.GROUP_CHAT,
        detected_routes=[],
        reason="open-ended questions require semantic coordination",
    )


def route_message(message: str) -> ConversationRoute:
    decision = decide_route(message)
    if len(decision.detected_routes) == 1:
        return decision.detected_routes[0]
    return ConversationRoute.RETRIEVAL

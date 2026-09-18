from __future__ import annotations

from enum import StrEnum


class ConversationRoute(StrEnum):
    RETRIEVAL = "retrieval"
    CSV = "csv"
    YIELD_PREDICTION = "yield_prediction"
    FE_PREDICTION = "fe_prediction"
    CATALYST_RECOMMENDATION = "catalyst_recommendation"


def route_message(message: str) -> ConversationRoute:
    text = message.casefold()
    if "recommend" in text or "推荐" in text:
        return ConversationRoute.CATALYST_RECOMMENDATION
    if "predict" in text and "yield" in text:
        return ConversationRoute.YIELD_PREDICTION
    if "predict" in text and ("faradaic" in text or " fe " in f" {text} "):
        return ConversationRoute.FE_PREDICTION
    if "csv" in text or "dataset" in text:
        return ConversationRoute.CSV
    return ConversationRoute.RETRIEVAL

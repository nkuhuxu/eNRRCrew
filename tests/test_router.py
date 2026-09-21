from enrrcrew.agents import (
    ConversationRoute,
    RouteStrategy,
    decide_route,
    route_message,
)


def test_deterministic_routes() -> None:
    assert route_message("predict yield for Fe-N-C") is ConversationRoute.YIELD_PREDICTION
    assert route_message("predict Faradaic efficiency") is ConversationRoute.FE_PREDICTION
    assert route_message("summarize this CSV") is ConversationRoute.CSV
    assert route_message("Which catalysts work in acid?") is ConversationRoute.RETRIEVAL


def test_routes_catalyst_recommendation() -> None:
    assert route_message("recommend a catalyst") is ConversationRoute.CATALYST_RECOMMENDATION
    assert route_message("催化剂推荐") is ConversationRoute.CATALYST_RECOMMENDATION


def test_chinese_prediction_routes() -> None:
    assert route_message("预测 Fe-N-C 的氨产率") is ConversationRoute.YIELD_PREDICTION
    assert route_message("预测 Fe-N-C 的法拉第效率") is ConversationRoute.FE_PREDICTION
    assert route_message("分析这个数据集") is ConversationRoute.CSV


def test_multiple_intents_use_group_chat() -> None:
    decision = decide_route("检索 FeMo 文献，再预测 FE")
    assert decision.strategy is RouteStrategy.GROUP_CHAT
    assert decision.detected_routes == [
        ConversationRoute.FE_PREDICTION,
        ConversationRoute.RETRIEVAL,
    ]


def test_single_explicit_intent_uses_fast_path() -> None:
    decision = decide_route("检索 FeMo 的研究")
    assert decision.strategy is RouteStrategy.FAST_PATH
    assert decision.detected_routes == [ConversationRoute.RETRIEVAL]


def test_open_question_uses_group_chat() -> None:
    decision = decide_route("Which catalyst works in acid?")
    assert decision.strategy is RouteStrategy.GROUP_CHAT
    assert decision.detected_routes == []


def test_research_does_not_accidentally_match_search_keyword() -> None:
    decision = decide_route("Explain current Fe-N-C research")
    assert decision.strategy is RouteStrategy.GROUP_CHAT
    assert decision.detected_routes == []

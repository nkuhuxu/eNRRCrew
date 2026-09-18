from enrrcrew.agents import ConversationRoute, route_message


def test_deterministic_routes() -> None:
    assert route_message("predict yield for Fe-N-C") is ConversationRoute.YIELD_PREDICTION
    assert route_message("predict Faradaic efficiency") is ConversationRoute.FE_PREDICTION
    assert route_message("summarize this CSV") is ConversationRoute.CSV
    assert route_message("Which catalysts work in acid?") is ConversationRoute.RETRIEVAL


def test_routes_catalyst_recommendation() -> None:
    assert route_message("recommend a catalyst") is ConversationRoute.CATALYST_RECOMMENDATION
    assert route_message("催化剂推荐") is ConversationRoute.CATALYST_RECOMMENDATION

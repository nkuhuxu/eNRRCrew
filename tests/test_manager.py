import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from enrrcrew.agents import (
    AgentArtifacts,
    AgentManager,
    AgentRequest,
    ConversationRoute,
    RouteStrategy,
)
from enrrcrew.agents.tools import AgentToolbox
from enrrcrew.schemas import (
    PredictionResult,
    PredictionType,
    RecommendationRequest,
    RecommendationResult,
)
from enrrcrew.services.conversation import ConversationService


class FakeRag:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def search(self, *args: object) -> str:
        self.calls.append(args)
        return "grounded answer"


def context(tmp_path: Path, *, api_key: str = "key") -> SimpleNamespace:
    settings = SimpleNamespace(
        api_key=api_key,
        base_url="https://example.test/v1",
        chat_model="gpt-4o-mini",
        manager_model="gpt-4o",
        agent_timeout=180,
        agent_max_rounds=8,
        asset_root=tmp_path / "assets",
        runtime_root=tmp_path / "runtime",
    )
    return SimpleNamespace(
        settings=settings,
        workspace=SimpleNamespace(root=tmp_path / "runtime" / "sessions" / "one"),
        rag=FakeRag(),
    )


def test_agent_manager_runs_explicit_retrieval_fast_path(tmp_path: Path) -> None:
    supplied = context(tmp_path)
    result = AgentManager().answer(
        AgentRequest(question="retrieve acid catalysts"), supplied
    )

    assert result.answer == "grounded answer"
    assert result.strategy is RouteStrategy.FAST_PATH
    assert supplied.rag.calls == [
        ("retrieve acid catalysts", "local", 0, "single paragraph")
    ]
    assert [event.event_type for event in result.trace] == ["tool_call", "tool_result"]


def test_agent_manager_requires_api_key(tmp_path: Path) -> None:
    result = AgentManager().answer(
        AgentRequest(question="Which catalyst works in acid?"),
        context(tmp_path, api_key=""),
    )
    assert result.status == "failed"
    assert "Configure an API key" in result.answer


def prediction_result(kind: PredictionType) -> PredictionResult:
    return PredictionResult(
        prediction_type=kind,
        electrocatalyst="Fe-N-C",
        category="High",
        probability=0.75 if kind is PredictionType.YIELD else None,
        cluster=2 if kind is PredictionType.FE else None,
        model_version="test",
        input_data={"applied_potential": -0.3},
    )


def recommendation_result() -> RecommendationResult:
    request = RecommendationRequest(allowed_elements={"Fe", "N", "C"})
    return RecommendationResult(
        request=request,
        recommendations=[],
        near_misses=[],
        rejected_counts={},
        model_versions={"yield": "test", "fe": "test"},
        run_id="run-one",
    )


@pytest.mark.parametrize(
    ("route", "method", "artifacts", "expected"),
    [
        (
            ConversationRoute.YIELD_PREDICTION,
            "predict_yield",
            AgentArtifacts(yield_result=prediction_result(PredictionType.YIELD)),
            "YIELD prediction",
        ),
        (
            ConversationRoute.FE_PREDICTION,
            "predict_fe",
            AgentArtifacts(fe_result=prediction_result(PredictionType.FE)),
            "FE prediction",
        ),
        (
            ConversationRoute.CATALYST_RECOMMENDATION,
            "recommend_catalysts",
            AgentArtifacts(recommendation_result=recommendation_result()),
            "No candidate satisfies",
        ),
        (
            ConversationRoute.CSV,
            "draft_csv_analysis",
            AgentArtifacts(csv_code="print('draft')"),
            "review the draft",
        ),
    ],
)
def test_fast_paths_format_typed_results(
    route: ConversationRoute,
    method: str,
    artifacts: AgentArtifacts,
    expected: str,
) -> None:
    toolbox = SimpleNamespace(
        artifacts=artifacts,
        trace=[],
        predict_yield=MagicMock(return_value=json.dumps({"status": "completed"})),
        predict_fe=MagicMock(return_value=json.dumps({"status": "completed"})),
        recommend_catalysts=MagicMock(
            return_value=json.dumps({"status": "completed"})
        ),
        draft_csv_analysis=MagicMock(
            return_value=json.dumps(
                {"status": "completed", "message": "review the draft"}
            )
        ),
    )

    result = AgentManager()._fast_path(
        AgentRequest(question="explicit task"), route, toolbox
    )

    assert result.status == "completed"
    assert expected in result.answer
    getattr(toolbox, method).assert_called_once()


def test_conversation_service_delegates_without_ui_state(tmp_path: Path) -> None:
    manager = MagicMock()
    expected = object()
    manager.answer.return_value = expected
    service = ConversationService(manager)
    request = AgentRequest(question="retrieve catalysts")
    supplied = context(tmp_path)

    assert service.answer(request, supplied) is expected
    manager.answer.assert_called_once_with(request, supplied)


def test_group_chat_is_bounded_and_uses_forced_final_summary(tmp_path: Path) -> None:
    supplied = context(tmp_path)

    class FakeCoordinator:
        @staticmethod
        def generate_reply(**kwargs):
            assert kwargs["messages"][-1]["role"] == "user"
            return "integrated answer"

    groupchat = SimpleNamespace(messages=[])

    class FakeUser:
        @staticmethod
        def initiate_chat(manager, **kwargs):
            assert kwargs["silent"] is True
            groupchat.messages.extend(
                [
                    {"name": "Retriever", "role": "assistant", "content": "evidence"},
                    {"name": "FE_Predictor", "role": "assistant", "content": "High"},
                ]
            )

    class TestManager(AgentManager):
        def _build_group_chat(self, toolbox, context):
            return FakeCoordinator(), groupchat, object(), [FakeUser()]

    result = TestManager().answer(
        AgentRequest(question="Which catalyst works in acid?"), supplied
    )
    assert result.answer == "integrated answer"
    assert result.strategy is RouteStrategy.GROUP_CHAT
    assert result.rounds == 3
    assert result.trace[-1].agent == "Coordinator"
    assert result.trace[-1].round_index == 8


def test_final_coordinator_never_receives_raw_unmatched_tool_calls(
    tmp_path: Path,
) -> None:
    supplied = context(tmp_path)

    class SafeCoordinator:
        @staticmethod
        def generate_reply(**kwargs):
            messages = kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "tool_calls" not in messages[0]
            assert "AUTHORITATIVE TOOL RESULTS" in messages[0]["content"]
            return "safe final answer"

    groupchat = SimpleNamespace(messages=[])

    class FakeUser:
        @staticmethod
        def initiate_chat(manager, **kwargs):
            groupchat.messages.append(
                {
                    "name": "Retriever",
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_evidence",
                                "arguments": '{"question":"Fe-N-C"}',
                            }
                        }
                    ],
                }
            )

    class SafeManager(AgentManager):
        def _build_group_chat(self, toolbox, context):
            return SafeCoordinator(), groupchat, object(), [FakeUser()]

    result = SafeManager().answer(
        AgentRequest(question="Explain Fe-N-C research"), supplied
    )

    assert result.status == "completed"
    assert result.answer == "safe final answer"


def test_group_chat_configuration_and_tool_ownership(tmp_path: Path) -> None:
    supplied = context(tmp_path)
    request = AgentRequest(question="Compare and predict")
    toolbox = AgentToolbox(request, supplied)
    _, groupchat, _, _ = AgentManager()._build_group_chat(toolbox, supplied)

    assert groupchat.speaker_selection_method == "auto"
    assert groupchat.allow_repeat_speaker is True
    assert groupchat.max_round == 7
    assert [agent.name for agent in groupchat.agents] == [
        "Coordinator",
        "Retriever",
        "Yield_Predictor",
        "FE_Predictor",
        "Recommendation_Specialist",
        "CSV_Analyst",
    ]
    assert groupchat.agents[0].function_map == {}
    assert set(groupchat.agents[1].function_map) == {"search_evidence"}
    assert set(groupchat.agents[2].function_map) == {"predict_yield"}
    assert set(groupchat.agents[3].function_map) == {"predict_fe"}
    assert set(groupchat.agents[4].function_map) == {"recommend_catalysts"}
    assert set(groupchat.agents[5].function_map) == {"draft_csv_analysis"}
    assert groupchat.agents[0].llm_config["config_list"][0]["model"] == "gpt-4o"
    assert all(
        agent.llm_config["config_list"][0]["model"] == "gpt-4o-mini"
        for agent in groupchat.agents[1:]
    )
    assert all(agent.code_executor is None for agent in groupchat.agents)


def test_round_limit_and_unauthorized_tool_call_return_partial(tmp_path: Path) -> None:
    supplied = context(tmp_path)

    class FakeCoordinator:
        @staticmethod
        def generate_reply(**kwargs):
            return "Only the completed evidence can be reported."

    groupchat = SimpleNamespace(messages=[])

    class FakeUser:
        @staticmethod
        def initiate_chat(manager, **kwargs):
            groupchat.messages.extend(
                [
                    {
                        "name": "Retriever",
                        "role": "assistant",
                        "content": "attempt",
                        "function_call": {"name": "predict_fe", "arguments": "{}"},
                    },
                    *[
                        {"name": "Coordinator", "role": "assistant", "content": f"round {i}"}
                        for i in range(6)
                    ],
                ]
            )

    class TestManager(AgentManager):
        def _build_group_chat(self, toolbox, context):
            return FakeCoordinator(), groupchat, object(), [FakeUser()]

    result = TestManager().answer(
        AgentRequest(question="Search and predict FE"), supplied
    )

    assert result.status == "partial"
    assert result.rounds == 8
    assert any(event.status == "rejected" for event in result.trace)
    assert result.trace[-1].agent == "Coordinator"


def test_manager_failure_is_explicit_and_does_not_fallback(tmp_path: Path) -> None:
    class FailedManager(AgentManager):
        def _build_group_chat(self, toolbox, context):
            raise RuntimeError("manager model unavailable")

    result = FailedManager().answer(
        AgentRequest(question="Explain how catalyst properties interact"),
        context(tmp_path),
    )

    assert result.status == "failed"
    assert result.answer == (
        "Multi-agent coordination failed (RuntimeError: manager model unavailable)."
    )
    assert result.trace[-1].agent == "Coordinator"


def test_coordinator_failure_preserves_verified_predictions(tmp_path: Path) -> None:
    supplied = context(tmp_path)

    class FailedCoordinator:
        @staticmethod
        def generate_reply(**kwargs):
            raise RuntimeError("temporary provider failure")

    groupchat = SimpleNamespace(messages=[])

    class FakeUser:
        @staticmethod
        def initiate_chat(manager, **kwargs):
            groupchat.messages.append(
                {"name": "Coordinator", "role": "assistant", "content": "working"}
            )

    class PartialManager(AgentManager):
        def _build_group_chat(self, toolbox, context):
            toolbox.artifacts.yield_result = prediction_result(PredictionType.YIELD)
            toolbox.artifacts.fe_result = prediction_result(PredictionType.FE)
            toolbox.results.update(
                {
                    "search_evidence": {
                        "status": "completed",
                        "evidence": "grounded evidence",
                    },
                    "predict_yield": {"status": "completed"},
                    "predict_fe": {"status": "completed"},
                }
            )
            return FailedCoordinator(), groupchat, object(), [FakeUser()]

    result = PartialManager().answer(
        AgentRequest(question="Retrieve evidence and predict yield and FE"), supplied
    )

    assert result.status == "partial"
    assert "YIELD prediction" in result.answer
    assert "FE prediction" in result.answer
    assert "temporary provider failure" in result.answer
    assert result.artifacts.yield_result is not None
    assert result.artifacts.fe_result is not None


def test_group_chat_propagates_needs_input_status(tmp_path: Path) -> None:
    supplied = context(tmp_path)

    class FakeCoordinator:
        @staticmethod
        def generate_reply(**kwargs):
            return "Please provide the catalyst elements and applied potential."

    groupchat = SimpleNamespace(messages=[])

    class FakeUser:
        @staticmethod
        def initiate_chat(manager, **kwargs):
            groupchat.messages.append(
                {"name": "FE_Predictor", "role": "assistant", "content": "Missing fields"}
            )

    class TestManager(AgentManager):
        def _build_group_chat(self, toolbox, context):
            toolbox.outcomes.append("needs_input")
            toolbox.results.update(
                {
                    "search_evidence": {"status": "completed"},
                    "predict_fe": {"status": "needs_input"},
                }
            )
            return FakeCoordinator(), groupchat, object(), [FakeUser()]

    result = TestManager().answer(
        AgentRequest(question="Retrieve evidence and predict FE"), supplied
    )

    assert result.status == "needs_input"
    assert "provide" in result.answer.lower()


def test_required_tool_completion_repairs_wrong_speaker_choice() -> None:
    toolbox = SimpleNamespace(
        results={
            "search_evidence": {"status": "completed"},
            "predict_yield": {"status": "completed"},
        },
        search_evidence=MagicMock(),
        predict_yield=MagicMock(),
        predict_fe=MagicMock(return_value='{"status":"completed"}'),
        recommend_catalysts=MagicMock(),
        draft_csv_analysis=MagicMock(),
    )
    request = AgentRequest(question="Retrieve evidence and predict yield and FE")

    AgentManager()._ensure_required_tools(
        request,
        [
            ConversationRoute.RETRIEVAL,
            ConversationRoute.YIELD_PREDICTION,
            ConversationRoute.FE_PREDICTION,
        ],
        toolbox,
    )

    toolbox.search_evidence.assert_not_called()
    toolbox.predict_yield.assert_not_called()
    toolbox.predict_fe.assert_called_once_with(request.question)

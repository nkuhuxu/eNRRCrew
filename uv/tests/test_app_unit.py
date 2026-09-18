from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from enrrcrew import app as app_module
from enrrcrew.schemas import (
    PredictionInput,
    PredictionResult,
    PredictionType,
    SandboxResult,
)


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def fake_streamlit() -> MagicMock:
    streamlit = MagicMock()
    streamlit.session_state = State()
    streamlit.sidebar = MagicMock()
    streamlit.sidebar.__enter__.return_value = streamlit.sidebar
    streamlit.columns.side_effect = lambda spec: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    streamlit.chat_message.return_value = MagicMock()
    streamlit.expander.return_value = MagicMock()
    streamlit.spinner.return_value = MagicMock()
    return streamlit


def prediction(prediction_type: PredictionType = PredictionType.YIELD) -> PredictionInput:
    return PredictionInput(
        prediction_type=prediction_type,
        applied_potential=-0.3,
        electrocatalyst="Fe-N-C",
        elements=["Fe", "N", "C"],
        morphology="porous nanosheet",
        ph_categories={"neutral"},
        electrolytes={"pbs"},
        n15_labeling=False,
    )


def result(prediction_type: PredictionType = PredictionType.YIELD) -> PredictionResult:
    return PredictionResult(
        prediction_type=prediction_type,
        electrocatalyst="Fe-N-C",
        category="High",
        probability=0.8 if prediction_type is PredictionType.YIELD else None,
        cluster=3 if prediction_type is PredictionType.FE else None,
        model_version="test-model",
        input_data=prediction(prediction_type).model_dump(mode="json"),
        warnings=["review"],
    )


def test_predictor_factory_session_and_sidebar(monkeypatch, tmp_path: Path) -> None:
    streamlit = fake_streamlit()
    monkeypatch.setattr(app_module, "st", streamlit)
    yield_marker = object()
    fe_marker = object()
    monkeypatch.setattr(app_module, "YieldPredictor", lambda path: yield_marker)
    monkeypatch.setattr(app_module, "FEPredictor", lambda path: fe_marker)
    factory = getattr(app_module.get_predictor, "__wrapped__", app_module.get_predictor)
    assert factory("yield", str(tmp_path)) is yield_marker
    assert factory("fe", str(tmp_path)) is fe_marker

    calls: list[str | None] = []

    def create(settings, session_id=None):
        calls.append(session_id)
        return SimpleNamespace(session_id=session_id or "new-session", root=tmp_path)

    monkeypatch.setattr(app_module.SessionWorkspace, "create", create)
    settings = SimpleNamespace(
        api_key="",
        base_url="https://example.test/v1",
        asset_root=tmp_path,
        validate_assets=lambda: ["missing-model"],
        with_session_credentials=MagicMock(return_value="updated"),
    )
    workspace = app_module.initialize_session(settings)
    assert workspace.session_id == "new-session"
    assert calls == [None, "new-session"]
    streamlit.text_input.side_effect = ["temporary-key", "https://example.test/v1"]
    assert app_module.render_sidebar(settings, workspace) == "updated"
    streamlit.error.assert_called_once()


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        (app_module.ConversationRoute.YIELD_PREDICTION, "Yield prediction"),
        (app_module.ConversationRoute.FE_PREDICTION, "FE prediction"),
        (app_module.ConversationRoute.CSV, "CSV analysis"),
        (app_module.ConversationRoute.CATALYST_RECOMMENDATION, "Catalyst recommendation"),
    ],
)
def test_chat_deterministic_routes(monkeypatch, route, expected: str) -> None:
    streamlit = fake_streamlit()
    streamlit.chat_input.return_value = "route this"
    monkeypatch.setattr(app_module, "st", streamlit)
    monkeypatch.setattr(app_module, "route_message", lambda prompt: route)
    app_module.render_chat(SimpleNamespace())
    assert expected in streamlit.session_state["chat_messages"][-1]["content"]


def test_chat_empty_success_and_failure_retrieval(monkeypatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        asset_root=tmp_path,
        graphrag_config=tmp_path / "settings.yaml",
        api_key="key",
        base_url="https://example.test/v1",
    )
    streamlit = fake_streamlit()
    monkeypatch.setattr(app_module, "st", streamlit)
    streamlit.chat_input.return_value = None
    app_module.render_chat(settings)
    assert streamlit.session_state["chat_messages"] == []

    streamlit.chat_input.return_value = "retrieve"
    monkeypatch.setattr(
        app_module, "route_message", lambda prompt: app_module.ConversationRoute.RETRIEVAL
    )
    managed = SimpleNamespace(content="evidence", debug_messages=("trace",))
    monkeypatch.setattr(
        app_module,
        "AgentManager",
        lambda service: SimpleNamespace(answer=lambda *args: managed),
    )
    app_module.render_chat(settings)
    assert streamlit.session_state["chat_messages"][-1]["content"] == "evidence"

    monkeypatch.setattr(
        app_module, "AgentManager", lambda service: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    app_module.render_chat(settings)
    assert "Retrieval failed" in streamlit.session_state["chat_messages"][-1]["content"]


def test_prediction_rendering_success_and_errors(monkeypatch, tmp_path: Path) -> None:
    streamlit = fake_streamlit()
    monkeypatch.setattr(app_module, "st", streamlit)
    app_module.render_header()
    app_module.render_result(result())
    assert streamlit.metric.call_count == 0

    value = prediction()
    streamlit.text_area.return_value = "Fe-N-C description"
    streamlit.button.return_value = True
    monkeypatch.setattr(app_module, "render_prediction_form", lambda kind: (True, value))
    monkeypatch.setattr(app_module, "apply_prediction_defaults", MagicMock())
    monkeypatch.setattr(
        app_module,
        "PredictionTextExtractor",
        lambda *args: SimpleNamespace(extract=lambda *args: value),
    )
    monkeypatch.setattr(
        app_module,
        "get_predictor",
        lambda *args: SimpleNamespace(predict=lambda supplied: result()),
    )
    settings = SimpleNamespace(
        api_key="key", base_url="https://example.test/v1", chat_model="model", models_dir=tmp_path
    )
    app_module.render_prediction_tab(settings, PredictionType.YIELD)
    assert streamlit.session_state["result_yield"]["category"] == "High"

    monkeypatch.setattr(
        app_module,
        "PredictionTextExtractor",
        lambda *args: SimpleNamespace(
            extract=lambda *args: (_ for _ in ()).throw(RuntimeError("bad extraction"))
        ),
    )
    monkeypatch.setattr(
        app_module,
        "get_predictor",
        lambda *args: SimpleNamespace(
            predict=lambda supplied: (_ for _ in ()).throw(RuntimeError("bad model"))
        ),
    )
    app_module.render_prediction_tab(settings, PredictionType.YIELD)
    assert streamlit.error.call_count >= 2


def test_upload_and_csv_analysis_paths(monkeypatch, tmp_path: Path) -> None:
    upload = SimpleNamespace(name="../dataset.csv", getvalue=lambda: b"x\n1\n")
    workspace = SimpleNamespace(root=tmp_path)
    saved = app_module._save_upload(workspace, upload)
    assert saved == tmp_path / "dataset.csv"
    with pytest.raises(ValueError, match="Only CSV"):
        app_module._save_upload(
            workspace, SimpleNamespace(name="payload.py", getvalue=lambda: b"pass")
        )

    streamlit = fake_streamlit()
    streamlit.radio.return_value = "Curated eNRR dataset"
    streamlit.text_area.side_effect = ["summarize", "print('ok')"]
    streamlit.button.side_effect = [True, True]
    monkeypatch.setattr(app_module, "st", streamlit)
    monkeypatch.setattr(
        app_module,
        "CsvAnalysisAgent",
        lambda *args: SimpleNamespace(generate_code=lambda *args: "print('ok')"),
    )
    sandbox_result = SandboxResult(
        exit_code=0,
        stdout="ok",
        stderr="",
        output_files=[tmp_path / "chart.png"],
        timed_out=False,
    )
    monkeypatch.setattr(
        app_module,
        "SandboxRunner",
        lambda *args: SimpleNamespace(run=lambda *args: sandbox_result),
    )
    settings = SimpleNamespace(
        input_dir=tmp_path,
        api_key="key",
        base_url="https://example.test/v1",
        chat_model="model",
        sandbox_image="image",
        sandbox_timeout=1,
    )
    app_module.render_csv_analysis(settings, workspace)
    assert streamlit.session_state["sandbox_result"]["exit_code"] == 0
    streamlit.image.assert_called_once()

    streamlit.text_area.side_effect = ["summarize", "print('ok')"]
    streamlit.button.side_effect = [True, True]
    monkeypatch.setattr(
        app_module,
        "CsvAnalysisAgent",
        lambda *args: SimpleNamespace(
            generate_code=lambda *args: (_ for _ in ()).throw(RuntimeError("LLM offline"))
        ),
    )
    monkeypatch.setattr(
        app_module,
        "SandboxRunner",
        lambda *args: SimpleNamespace(
            run=lambda *args: (_ for _ in ()).throw(RuntimeError("Docker offline"))
        ),
    )
    app_module.render_csv_analysis(settings, workspace)
    assert streamlit.error.call_count >= 2


def test_main_dispatches_all_workspaces(monkeypatch, tmp_path: Path) -> None:
    streamlit = fake_streamlit()
    streamlit.tabs.return_value = [MagicMock() for _ in range(5)]
    streamlit.session_state["pending_prediction"] = prediction().model_dump(mode="json")
    monkeypatch.setattr(app_module, "st", streamlit)
    settings = SimpleNamespace()
    workspace = SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(app_module.AppSettings, "from_environment", lambda: settings)
    monkeypatch.setattr(app_module, "initialize_session", lambda supplied: workspace)
    monkeypatch.setattr(app_module, "render_sidebar", lambda *args: settings)
    monkeypatch.setattr(app_module, "apply_theme", MagicMock())
    monkeypatch.setattr(app_module, "apply_prediction_defaults", MagicMock())
    monkeypatch.setattr(app_module, "render_header", MagicMock())
    monkeypatch.setattr(app_module, "render_chat", MagicMock())
    monkeypatch.setattr(app_module, "render_prediction_tab", MagicMock())
    monkeypatch.setattr(app_module, "render_csv_analysis", MagicMock())
    monkeypatch.setattr(app_module, "render_recommendation", MagicMock())
    app_module.main()
    assert app_module.render_prediction_tab.call_count == 2
    app_module.render_recommendation.assert_called_once_with(settings, workspace)

from pathlib import Path

from streamlit.testing.v1 import AppTest

from enrrcrew.schemas import PredictionInput, RecommendationResult
from enrrcrew.services import PredictionTextExtractor
from enrrcrew.ui import recommendation as recommendation_ui

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "enrrcrew" / "app.py"
ASSET_ROOT = Path(__file__).resolve().parents[2]


def app(monkeypatch, tmp_path: Path, *, api_key: str = "") -> AppTest:
    monkeypatch.setenv("ENRRCREW_ASSET_ROOT", str(ASSET_ROOT))
    monkeypatch.setenv("ENRRCREW_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("GRAPHRAG_API_KEY", api_key)
    return AppTest.from_file(str(APP_PATH), default_timeout=90).run()


def button_with_label(application: AppTest, label: str, index: int = 0):
    matches = [button for button in application.button if button.label == label]
    return matches[index]


def test_chat_history_and_safe_retrieval_error(monkeypatch, tmp_path: Path) -> None:
    application = app(monkeypatch, tmp_path)
    application.chat_input[0].set_value("Which catalyst works in acid?").run()

    assert len(application.chat_message) == 2
    assert "Configure an API key" in application.chat_message[1].markdown[0].value
    assert not any(expander.label == "Agent trace" for expander in application.expander)


def test_prediction_form_displays_validation_error(monkeypatch, tmp_path: Path) -> None:
    application = app(monkeypatch, tmp_path)
    application.text_input(key="prediction_yield_catalyst").set_value("test")
    application.text_input(key="prediction_yield_elements").set_value("Xx")
    button_with_label(application, "Validate & predict", 0).click().run()

    assert any("Please correct" in error.value for error in application.error)
    assert any("unknown element symbol" in caption.value for caption in application.caption)


def test_extracted_input_can_be_corrected_before_prediction(
    monkeypatch, tmp_path: Path
) -> None:
    extracted = PredictionInput.model_validate(
        {
            "prediction_type": "yield",
            "applied_potential": -0.3,
            "electrocatalyst": "Extracted catalyst",
            "elements": ["Fe", "N", "C"],
            "morphology": "porous nanosheet",
            "ph_categories": ["neutral"],
            "electrolytes": ["pbs"],
            "n15_labeling": True,
        }
    )
    monkeypatch.setattr(
        PredictionTextExtractor,
        "extract",
        lambda self, text, prediction_type: extracted,
    )
    application = app(monkeypatch, tmp_path, api_key="session-test-key")
    application.text_area(key="natural_yield").set_value("structured experiment")
    button_with_label(application, "Extract & review").click().run()

    catalyst = application.text_input(key="prediction_yield_catalyst")
    assert catalyst.value == "Extracted catalyst"
    catalyst.set_value("Corrected Fe-N-C")
    button_with_label(application, "Validate & predict", 0).click().run(timeout=90)

    assert any(metric.label == "Classification" for metric in application.metric)
    assert "Corrected Fe-N-C" in str(application.session_state["result_yield"])


def test_recommendation_form_submits_without_llm(monkeypatch, tmp_path: Path) -> None:
    class FakeRecommendationService:
        @staticmethod
        def default_allowed_elements() -> list[str]:
            return ["Fe", "N", "C"]

        @staticmethod
        def recommend(request):
            return RecommendationResult(
                request=request,
                recommendations=[],
                near_misses=[],
                rejected_counts={
                    "evaluated_known": 3,
                    "generated": 0,
                    "strict_double_high": 0,
                    "out_of_domain": 0,
                },
                model_versions={"yield": "test", "fe": "test"},
                run_id="test-run",
            )

    monkeypatch.setattr(
        recommendation_ui,
        "get_recommendation_service",
        lambda *args: FakeRecommendationService(),
    )
    application = app(monkeypatch, tmp_path)
    button_with_label(application, "Run recommendation").click().run()

    assert not application.exception
    assert any(metric.label == "Known evaluated" for metric in application.metric)
    assert any("No candidate satisfies" in warning.value for warning in application.warning)

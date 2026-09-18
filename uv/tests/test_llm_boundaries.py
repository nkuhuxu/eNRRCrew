import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from enrrcrew.agents import csv_analyst
from enrrcrew.schemas import PredictionType
from enrrcrew.services import text_extraction


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return _response(self.content)


class _FakeOpenAI:
    content = ""
    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs: object):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions(self.content))
        self.instances.append(self)


def test_text_extractor_validates_json_and_forces_requested_type(monkeypatch) -> None:
    payload = {
        "prediction_type": "fe",
        "applied_potential": -0.45,
        "electrocatalyst": "Fe-N-C",
        "elements": ["Fe", "N", "C"],
        "morphology": "porous nanosheet",
        "ph_categories": ["neutral"],
        "electrolytes": ["pbs"],
        "n15_labeling": True,
    }
    _FakeOpenAI.content = json.dumps(payload)
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(text_extraction, "OpenAI", _FakeOpenAI)

    extractor = text_extraction.PredictionTextExtractor(
        "session-key", "https://example.test/v1", "test-model"
    )
    result = extractor.extract("Fe-N-C at -0.45 V in PBS", PredictionType.YIELD)

    assert result.prediction_type is PredictionType.YIELD
    assert result.elements == ["Fe", "N", "C"]
    client = _FakeOpenAI.instances[-1]
    assert client.kwargs["api_key"] == "session-key"
    call = client.chat.completions.calls[-1]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("text", ["", "   "])
def test_text_extractor_rejects_empty_input(monkeypatch, text: str) -> None:
    monkeypatch.setattr(text_extraction, "OpenAI", _FakeOpenAI)
    extractor = text_extraction.PredictionTextExtractor("key", "https://test/v1", "model")
    with pytest.raises(ValueError, match="cannot be empty"):
        extractor.extract(text, PredictionType.FE)


def test_csv_agent_includes_columns_and_removes_markdown_fence(
    monkeypatch, tmp_path: Path
) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("catalyst,fe\nFe-N-C,82\n", encoding="utf-8")
    _FakeOpenAI.content = "```python\nimport pandas as pd\nprint('ok')\n```"
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(csv_analyst, "OpenAI", _FakeOpenAI)

    agent = csv_analyst.CsvAnalysisAgent("session-key", "https://test/v1", "model")
    code = agent.generate_code("Summarize FE", csv_path, "dataset.csv")

    assert code == "import pandas as pd\nprint('ok')"
    prompt = _FakeOpenAI.instances[-1].chat.completions.calls[-1]["messages"][1][
        "content"
    ]
    assert "['catalyst', 'fe']" in prompt
    assert "/data/dataset.csv" in prompt


def test_llm_features_require_session_credentials() -> None:
    with pytest.raises(ValueError, match="API key"):
        text_extraction.PredictionTextExtractor("", "https://test/v1", "model")
    with pytest.raises(ValueError, match="API key"):
        csv_analyst.CsvAnalysisAgent("", "https://test/v1", "model")

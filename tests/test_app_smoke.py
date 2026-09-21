from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_with_six_workspaces(monkeypatch, tmp_path: Path) -> None:
    asset_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("ENRRCREW_ASSET_ROOT", str(asset_root))
    monkeypatch.setenv("ENRRCREW_RUNTIME_ROOT", str(tmp_path / "runtime"))
    app_path = Path(__file__).resolve().parents[1] / "src" / "enrrcrew" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    assert not app.exception
    assert len(app.tabs) == 6
    assert app.tabs[-1].label.endswith("Knowledge base update")

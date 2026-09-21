import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from enrrcrew.services import SandboxRunner, validate_code


@pytest.mark.parametrize(
    "code",
    [
        "import os\nprint(os.environ)",
        "import subprocess\nsubprocess.run(['whoami'])",
        "open('/etc/passwd').read()",
        "getattr(object(), '__class__')",
        "import pandas as pd\nprint(pd.__dict__)",
        "import pandas as pd\nprint(pd.read_csv('/etc/passwd'))",
        "import pandas as pd\nprint(pd.read_csv('/data/../etc/passwd'))",
        r"import pandas as pd; pd.read_csv('C:\\Users\\secret.csv')",
    ],
)
def test_validator_rejects_dangerous_code(code: str) -> None:
    with pytest.raises(ValueError):
        validate_code(code)


def test_validator_accepts_bounded_analysis() -> None:
    validate_code(
        "import pandas as pd\n"
        "df = pd.read_csv('/data/dataset.csv')\n"
        "print(df.describe().to_string())\n"
    )


@pytest.mark.parametrize("code", ["", "if broken syntax"])
def test_validator_rejects_empty_or_invalid_syntax(code: str) -> None:
    with pytest.raises(ValueError):
        validate_code(code)


def test_runner_does_not_fall_back_when_docker_is_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    from docker.errors import DockerException

    from enrrcrew.services import sandbox as sandbox_module

    dataset = tmp_path / "dataset.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(
        sandbox_module.docker,
        "from_env",
        lambda: (_ for _ in ()).throw(DockerException("offline")),
    )
    with pytest.raises(RuntimeError, match="local execution is intentionally disabled"):
        SandboxRunner("test-image").run(
            "import pandas as pd\nprint(pd.read_csv('/data/dataset.csv'))",
            {"dataset.csv": dataset},
            tmp_path,
        )


def test_container_timeout_is_reported(monkeypatch, tmp_path: Path) -> None:
    from enrrcrew.services import sandbox as sandbox_module

    class FakeContainer:
        killed = False

        def exec_run(self, command: list[str]) -> SimpleNamespace:
            return SimpleNamespace(exit_code=1, output=b"")

        def kill(self) -> None:
            self.killed = True

        def logs(self, *, stdout: bool, stderr: bool) -> bytes:
            return b""

        def remove(self, *, force: bool) -> None:
            assert force is True

    container = FakeContainer()

    class FakeClient:
        containers = type(
            "Containers", (), {"run": staticmethod(lambda *args, **kwargs: container)}
        )()

        def ping(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(sandbox_module.docker, "from_env", FakeClient)

    result = SandboxRunner("test-image", timeout=0).run(
        "print('bounded')", {}, tmp_path
    )

    assert result.timed_out is True
    assert result.exit_code == 124
    assert container.killed is True


def test_output_manifest_rejects_unsafe_paths(tmp_path: Path) -> None:
    class FakeContainer:
        def exec_run(self, command: list[str]) -> SimpleNamespace:
            manifest = json.dumps([{"path": "../escape.txt", "size": 6}]).encode()
            return SimpleNamespace(exit_code=0, output=manifest)

    runner = SandboxRunner("test-image")
    with pytest.raises(RuntimeError, match="unsafe"):
        runner._copy_outputs(FakeContainer(), tmp_path.resolve())


def test_output_archive_enforces_total_size(tmp_path: Path) -> None:
    class FakeContainer:
        def exec_run(self, command: list[str]) -> SimpleNamespace:
            manifest = json.dumps([{"path": "large.bin", "size": 5}]).encode()
            return SimpleNamespace(exit_code=0, output=manifest)

    with pytest.raises(RuntimeError, match="exceeded"):
        SandboxRunner("test-image", max_artifact_bytes=4)._copy_outputs(
            FakeContainer(), tmp_path.resolve()
        )


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (None, "did not produce"),
        ({"path": "chart.png"}, "invalid output manifest"),
        (["chart.png"], "invalid output manifest entry"),
        ([{"path": "/absolute.png", "size": 1}], "unsafe output path"),
        ([{"path": "negative.png", "size": -1}], "unsafe output path"),
    ],
)
def test_output_manifest_validation(manifest, message: str, tmp_path: Path) -> None:
    class FakeContainer:
        def exec_run(self, command: list[str]) -> SimpleNamespace:
            if manifest is None:
                return SimpleNamespace(exit_code=1, output=b"")
            return SimpleNamespace(exit_code=0, output=json.dumps(manifest).encode())

    with pytest.raises(RuntimeError, match=message):
        SandboxRunner("test-image")._copy_outputs(FakeContainer(), tmp_path.resolve())


def test_output_copy_checks_size_and_writes_only_inside_session(tmp_path: Path) -> None:
    class FakeContainer:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def exec_run(self, command: list[str]) -> SimpleNamespace:
            if command[-1] == "/tmp/analysis-outputs.json":
                manifest = [{"path": "plots/chart.png", "size": 3}]
                return SimpleNamespace(exit_code=0, output=json.dumps(manifest).encode())
            return SimpleNamespace(exit_code=0, output=self.payload)

    runner = SandboxRunner("test-image")
    with pytest.raises(RuntimeError, match="size did not match"):
        runner._copy_outputs(FakeContainer(b"xx"), tmp_path.resolve())

    copied = runner._copy_outputs(FakeContainer(b"png"), tmp_path.resolve())
    assert copied == [tmp_path / "plots" / "chart.png"]
    assert copied[0].read_bytes() == b"png"


def test_runner_rejects_unsafe_alias_and_missing_input(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")
    runner = SandboxRunner("test-image")
    with pytest.raises(ValueError, match="plain filename"):
        runner.run("print('safe')", {"../dataset.csv": dataset}, tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.run("print('safe')", {"dataset.csv": tmp_path / "missing.csv"}, tmp_path)


def test_successful_runner_enforces_container_security(monkeypatch, tmp_path: Path) -> None:
    from enrrcrew.services import sandbox as sandbox_module

    dataset = tmp_path / "dataset.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")

    class FakeContainer:
        removed = False

        def exec_run(self, command: list[str]) -> SimpleNamespace:
            path = command[-1]
            if path == "/tmp/analysis-exit-code":
                return SimpleNamespace(exit_code=0, output=b"0\n")
            if path == "/tmp/analysis-outputs.json":
                value = [{"path": "summary.txt", "size": 2}]
                return SimpleNamespace(exit_code=0, output=json.dumps(value).encode())
            return SimpleNamespace(exit_code=0, output=b"ok")

        def logs(self, *, stdout: bool, stderr: bool) -> bytes:
            return b"stdout-long" if stdout else b"stderr-long"

        def remove(self, *, force: bool) -> None:
            self.removed = force

    container = FakeContainer()
    run_options = {}

    class Containers:
        @staticmethod
        def run(*args, **kwargs):
            run_options.update(kwargs)
            return container

    class FakeClient:
        containers = Containers()
        closed = False

        def ping(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    client = FakeClient()
    monkeypatch.setattr(sandbox_module.docker, "from_env", lambda: client)
    output = SandboxRunner("test-image", max_output_bytes=6).run(
        "print('safe')", {"dataset.csv": dataset}, tmp_path
    )

    assert output.stdout == "stdout"
    assert output.stderr == "stderr"
    assert output.output_files == [tmp_path / "output" / "summary.txt"]
    assert run_options["network_disabled"] is True
    assert run_options["read_only"] is True
    assert run_options["cap_drop"] == ["ALL"]
    assert run_options["pids_limit"] == 64
    assert container.removed is True
    assert client.closed is True


def test_missing_sandbox_image_is_explicit(monkeypatch, tmp_path: Path) -> None:
    from docker.errors import ImageNotFound

    from enrrcrew.services import sandbox as sandbox_module

    class Containers:
        @staticmethod
        def run(*args, **kwargs):
            raise ImageNotFound("missing")

    class FakeClient:
        containers = Containers()

        def ping(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(sandbox_module.docker, "from_env", FakeClient)
    with pytest.raises(RuntimeError, match="image 'missing-image' is missing"):
        SandboxRunner("missing-image").run("print('safe')", {}, tmp_path)

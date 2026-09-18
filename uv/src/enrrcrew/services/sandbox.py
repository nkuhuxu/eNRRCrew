from __future__ import annotations

import ast
import json
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from docker.errors import DockerException, ImageNotFound

import docker
from enrrcrew.schemas import SandboxResult

ALLOWED_IMPORTS = {
    "csv",
    "json",
    "math",
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn",
    "statistics",
}
BLOCKED_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}


def validate_code(code: str) -> None:
    if not code.strip():
        raise ValueError("Analysis code cannot be empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            posix = PurePosixPath(value)
            windows = PureWindowsPath(value)
            if windows.drive or value.startswith("\\\\"):
                raise ValueError("Host filesystem paths are not allowed in the sandbox")
            if value.startswith("/"):
                allowed_root = len(posix.parts) >= 3 and posix.parts[1] in {
                    "data",
                    "output",
                }
                if not allowed_root or ".." in posix.parts:
                    raise ValueError(f"Filesystem path is outside sandbox mounts: {value}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".", maxsplit=1)[0]
                if root not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import is not allowed in the sandbox: {name}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BLOCKED_CALLS
        ):
            raise ValueError(f"Call is not allowed in the sandbox: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private and dunder attribute access is not allowed")


class SandboxRunner:
    def __init__(
        self,
        image: str,
        timeout: int = 30,
        max_output_bytes: int = 100_000,
        max_artifact_bytes: int = 64 * 1024 * 1024,
    ):
        self.image = image
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes
        self.max_artifact_bytes = max_artifact_bytes

    def _copy_outputs(self, container: object, output_dir: Path) -> list[Path]:
        manifest_result = container.exec_run(["cat", "/tmp/analysis-outputs.json"])
        if manifest_result.exit_code != 0:
            raise RuntimeError("Sandbox did not produce an output manifest")
        manifest = json.loads(manifest_result.output.decode("utf-8"))
        if not isinstance(manifest, list):
            raise RuntimeError("Sandbox returned an invalid output manifest")
        copied: list[Path] = []
        total_size = 0
        for item in manifest:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise RuntimeError("Sandbox returned an invalid output manifest entry")
            relative = PurePosixPath(item["path"])
            declared_size = item.get("size")
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or not isinstance(declared_size, int)
                or declared_size < 0
            ):
                raise RuntimeError("Sandbox returned an unsafe output path")
            total_size += declared_size
            if total_size > self.max_artifact_bytes:
                raise RuntimeError("Sandbox artifacts exceeded the configured limit")

            file_result = container.exec_run(
                ["cat", f"/output/{relative.as_posix()}"]
            )
            if file_result.exit_code != 0 or len(file_result.output) != declared_size:
                raise RuntimeError("Sandbox output size did not match its manifest")
            destination = output_dir.joinpath(*relative.parts).resolve()
            if not destination.is_relative_to(output_dir):
                raise RuntimeError("Sandbox output escaped the session directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file_result.output)
            copied.append(destination)
        return copied

    def _wait_for_completion(self, container: object) -> tuple[int, bool]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = container.exec_run(["cat", "/tmp/analysis-exit-code"])
            if result.exit_code == 0:
                return int(result.output.decode("ascii").strip()), False
            time.sleep(0.05)
        container.kill()
        return 124, True

    def run(
        self,
        code: str,
        input_files: dict[str, Path],
        workspace: Path,
    ) -> SandboxResult:
        validate_code(code)
        workspace = workspace.resolve()
        output_dir = workspace / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        code_path = workspace / f"analysis_{uuid4().hex}.py"
        code_path.write_text(code, encoding="utf-8")

        volumes: dict[str, dict[str, str]] = {
            str(code_path): {"bind": "/workspace/analysis.py", "mode": "ro"},
        }
        for alias, source in input_files.items():
            safe_alias = Path(alias).name
            if safe_alias != alias:
                raise ValueError(f"Input alias must be a plain filename: {alias}")
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            volumes[str(source)] = {"bind": f"/data/{safe_alias}", "mode": "ro"}

        try:
            client = docker.from_env()
            client.ping()
        except DockerException as exc:
            raise RuntimeError(
                "Docker is unavailable. Start Docker Desktop; local execution is intentionally disabled."
            ) from exc

        container = None
        timed_out = False
        try:
            container = client.containers.run(
                self.image,
                command=["python", "/sandbox_runner.py"],
                detach=True,
                working_dir="/workspace",
                network_disabled=True,
                read_only=True,
                volumes=volumes,
                tmpfs={
                    "/tmp": "rw,noexec,nosuid,size=64m",
                    "/output": (
                        "rw,noexec,nosuid,size=" f"{self.max_artifact_bytes}"
                    ),
                },
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit="512m",
                nano_cpus=1_000_000_000,
                pids_limit=64,
            )
            exit_code, timed_out = self._wait_for_completion(container)
            stdout = container.logs(stdout=True, stderr=False)
            stderr = container.logs(stdout=False, stderr=True)
            output_files = (
                []
                if timed_out or exit_code != 0
                else self._copy_outputs(container, output_dir)
            )
        except ImageNotFound as exc:
            raise RuntimeError(
                f"Sandbox image '{self.image}' is missing. Build docker/csv-sandbox first."
            ) from exc
        finally:
            if container is not None:
                container.remove(force=True)
            client.close()

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout[: self.max_output_bytes].decode("utf-8", errors="replace"),
            stderr=stderr[: self.max_output_bytes].decode("utf-8", errors="replace"),
            output_files=output_files,
            timed_out=timed_out,
        )

from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.services import SandboxRunner

settings = AppSettings.from_environment()
workspace = SessionWorkspace.create(settings, "docker-timeout-smoke")
result = SandboxRunner(settings.sandbox_image, timeout=1).run(
    "while True:\n    pass\n", {}, workspace.root
)
assert result.timed_out is True
assert result.exit_code == 124
print("Docker timeout enforcement passed")

from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.services import SandboxRunner

settings = AppSettings.from_environment()
workspace = SessionWorkspace.create(settings, "docker-output-limit-smoke")
code = """
import numpy as np

np.save('/output/too-large.npy', np.zeros(2_000_000, dtype=np.uint8))
"""
result = SandboxRunner(
    settings.sandbox_image, timeout=15, max_artifact_bytes=1024 * 1024
).run(code, {}, workspace.root)
assert result.exit_code != 0
assert not result.output_files
assert "OSError" in result.stderr
assert "requested" in result.stderr and "written" in result.stderr
print("Docker output capacity limit passed")

from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.services import SandboxRunner

settings = AppSettings.from_environment()
workspace = SessionWorkspace.create(settings, "docker-network-smoke")
code = """
import pandas as pd

try:
    pd.read_csv('http://example.com/network-must-be-disabled.csv')
except Exception:
    print('network_blocked')
else:
    raise RuntimeError('sandbox unexpectedly reached the network')
"""
result = SandboxRunner(settings.sandbox_image, timeout=10).run(code, {}, workspace.root)
assert result.exit_code == 0, result.stderr
assert result.stdout.strip() == "network_blocked"
print("Docker network isolation passed")

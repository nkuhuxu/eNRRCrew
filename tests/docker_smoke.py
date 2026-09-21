from enrrcrew.config import AppSettings, SessionWorkspace
from enrrcrew.services import SandboxRunner

settings = AppSettings.from_environment()
workspace = SessionWorkspace.create(settings, "docker-smoke")
dataset = settings.input_dir / "data_include_morphology_electrocatalyst.csv"
code = """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/data/dataset.csv')
print(f'rows={len(df)} columns={len(df.columns)}')
df['Faraday efficiency'].dropna().head(20).plot(kind='bar', color='#b8f34a')
plt.tight_layout()
plt.savefig('/output/sandbox-smoke.png')
print(f'png_bytes={len(np.fromfile("/output/sandbox-smoke.png", dtype=np.uint8))}')
"""
result = SandboxRunner(settings.sandbox_image, settings.sandbox_timeout).run(
    code, {"dataset.csv": dataset}, workspace.root
)
assert result.exit_code == 0, result.stderr
assert "rows=" in result.stdout
assert any(path.name == "sandbox-smoke.png" for path in result.output_files)
print(result.stdout.strip())
print("Docker sandbox smoke test passed")

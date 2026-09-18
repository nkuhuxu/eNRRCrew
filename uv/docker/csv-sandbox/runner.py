from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

result = subprocess.run(["python", "/workspace/analysis.py"], check=False)
outputs = [
    {"path": path.relative_to("/output").as_posix(), "size": path.stat().st_size}
    for path in Path("/output").rglob("*")
    if path.is_file() and not path.is_symlink()
]
Path("/tmp/analysis-outputs.json").write_text(json.dumps(outputs), encoding="utf-8")
Path("/tmp/analysis-exit-code").write_text(str(result.returncode), encoding="ascii")
while True:
    time.sleep(3600)

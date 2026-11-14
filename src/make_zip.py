"""
src/make_zip.py
Empaqueta el lliurament del Mode Difícil (agent.py + params.json) a submission_hard.zip.

Ús:
    python src/make_zip.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission_hard"
OUT = ROOT / "submission_hard.zip"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("agent.py", "params.json"):
        z.write(SRC / name, name)
print(f"Creat {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")

"""
src/make_zip.py
Empaqueta els lliuraments del Mode Difícil (agent.py + hard_v2_model.py + params.json):
  submission_hard.zip         -> v6 del Dani
  submission_hard_robust.zip  -> v6-robust (tau 20)

Ús:
    python src/make_zip.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("submission_hard", "submission_hard_robust"):
    SRC = ROOT / folder
    OUT = ROOT / f"{folder}.zip"
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("agent.py", "hard_v2_model.py", "params.json"):
            z.write(SRC / name, name)
    print(f"Creat {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")

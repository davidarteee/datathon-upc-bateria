"""Empaqueta el ZIP v2 (determinista) i el comprova amb el verificador oficial i un episodi aïllat.

    python -B experiments/dani_hard_v2/package.py
"""
import hashlib, json, subprocess, sys, tempfile, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "hard_v6_submission.zip"

ISOLATED = r'''
import sys, json, hashlib, importlib.util
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import agent
import numpy as np
controller = agent.BatteryAgent()
root = Path(sys.argv[2])
sys.path.insert(1, str(root))
from experiments.oriol_hard_mode import lab
frame = lab.load_hard()
env = lab.official_env()(frame, spec=lab.HARD_SPEC, export_price=0.0)
obs = lab.hard_observation(env.reset())
while obs is not None:
    action = controller.act(obs)
    assert isinstance(action, float) and np.isfinite(action)
    raw, _ = env.step(action); obs = lab.hard_observation(raw)
r = env.evaluate_results()
mods = {n: m.__file__ for n, m in sys.modules.items() if n in ("agent", "hard_v2_model")}
assert all(Path(p).resolve().is_relative_to(Path(sys.argv[1]).resolve()) for p in mods.values()), mods
print(json.dumps({"steps": len(frame), "score_train_in_sample": r["score"], "co2": r["co2_estalviat_pct"], "cost": r["cost_estalviat_pct"],
                  "infraccions": r["infraccions"], "failures": controller.policy_failures,
                  "actions_sha256": hashlib.sha256(np.asarray(env.actions, dtype="<f8").tobytes()).hexdigest()}))
'''

def main():
    payload = {"agent.py": (HERE / "agent.py").read_bytes(),
               "hard_v2_model.py": (HERE / "model.py").read_bytes(),
               "params.json": (HERE / "params.json").read_bytes()}
    if (HERE / "dashboard.html").exists():
        payload["dashboard.html"] = (HERE / "dashboard.html").read_bytes()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 14, 0, 0, 0)); entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3; entry.external_attr = 0o100644 << 16; archive.writestr(entry, content)
    print("zip", OUT.stat().st_size, "bytes sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
    checker = ROOT.parent / "paquet_participants_datathon_upc/starter_kit/check_submission.py"
    result = subprocess.run([sys.executable, "-I", "-B", str(checker), str(OUT)], capture_output=True, text=True)
    print(result.stdout[-900:], result.stderr[-500:]); assert result.returncode == 0
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(OUT) as z: z.extractall(tmp)
        result = subprocess.run([sys.executable, "-I", "-B", "-c", ISOLATED, tmp, str(ROOT)], cwd=tmp, capture_output=True, text=True, timeout=300)
        print(result.stdout, result.stderr[-1500:]); assert result.returncode == 0

if __name__ == "__main__":
    main()

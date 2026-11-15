"""agent.py — La MasIA, Mode Difícil, versió final (v10 sense buidat de l'últim dia)."""

import json
from pathlib import Path

from hard_v2_model import HardAgentV2, clamp_action


class BatteryAgent:
    def __init__(self):
        folder = Path(__file__).resolve().parent
        params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
        self.controller = HardAgentV2(params)

    @property
    def policy_failures(self):
        return self.controller.failures

    def act(self, obs: dict) -> float:
        try:
            return float(clamp_action(self.controller.act(obs), obs))
        except Exception:
            self.controller.failures += 1
            return 0.0

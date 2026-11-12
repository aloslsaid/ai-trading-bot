"""Configuration loading: YAML defaults + optional override file. No secrets live here."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@dataclass
class Settings:
    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def symbol(self) -> str:
        return self.raw["symbol"]

    @property
    def timeframes(self) -> list[str]:
        return list(self.raw["timeframes"])

    def path(self, key: str) -> Path:
        p = Path(self.raw["data"][key])
        p = p if p.is_absolute() else PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_settings(override: str | Path | None = None) -> Settings:
    with open(PROJECT_ROOT / "config" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(override, encoding="utf-8") as f:
            cfg = _deep_update(copy.deepcopy(cfg), yaml.safe_load(f) or {})
    return Settings(cfg)

"""YAML config loading with `a.b.c=value` command-line overrides."""
from pathlib import Path

import yaml


class AttrDict(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def to_attr(obj):
    if isinstance(obj, dict):
        return AttrDict({k: to_attr(v) for k, v in obj.items()})
    return obj


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    return obj


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like a.b.c=value, got '{item}'")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, overrides or [])
    return to_attr(cfg)

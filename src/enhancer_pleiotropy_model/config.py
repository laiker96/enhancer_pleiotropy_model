"""YAML configuration loading with ``${name}`` substitution and command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any

import yaml

_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _substitute(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, variables) for item in value]
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            name = match.group(1)
            if name not in variables:
                raise KeyError(f"Undefined configuration variable ${{{name}}}")
            return variables[name]
        return _REFERENCE.sub(replace, value)
    return value


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise KeyError(f"Cannot set {dotted}: {key} is not a mapping")
    node[keys[-1]] = value


def load_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML file, apply ``key.sub=value`` overrides, then resolve ``${var}``.

    Top-level string values can be referenced from anywhere as ``${name}``.
    Relative paths are interpreted relative to the current working directory.
    """
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{path}: configuration root must be a mapping")
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.subkey=value")
        key, text = item.split("=", 1)
        set_path(config, key.strip(), _parse_scalar(text))
    variables = {key: value for key, value in config.items() if isinstance(value, str)}
    for _ in range(10):  # allow variables that reference other variables
        resolved = {key: _substitute(value, variables) for key, value in variables.items()}
        if resolved == variables:
            break
        variables = resolved
    config = _substitute(copy.deepcopy(config), variables)
    config["_config_path"] = str(Path(path).resolve())
    return config


def without_private(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}

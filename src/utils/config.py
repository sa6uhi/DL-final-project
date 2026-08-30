"""Centralized configuration loader with dot-notation access.

All hyperparameters, paths, seeds, and thresholds must be sourced from the
single configuration file ``config/config.yaml`` through this module.
Relative paths declared inside the YAML file are resolved against the
repository root (the directory containing the ``config/`` folder), so
configurations remain portable between machines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
_MISSING = object()


def _resolve_base_dir(config_path: Path) -> Path:
    """Determine the directory absolute paths are resolved against.

    Paths are resolved relative to the repository root, i.e. the directory
    containing the ``config/`` folder. If the configuration file lives
    outside a ``config`` directory, its own directory is used instead.

    Args:
        config_path: Absolute path of the loaded configuration file.

    Returns:
        Directory used as the anchor for relative paths.
    """
    parent = config_path.parent
    return parent.parent if parent.name == "config" else parent


class Config(dict):
    """Recursive dictionary with dot-notation attribute access.

    Values that are themselves dictionaries are wrapped into nested
    :class:`Config` instances so that nested keys can be read either as
    ``config["paths"]["data"]`` or ``config.paths.data``.

    Examples:
        >>> cfg = load_config("config/config.yaml")
        >>> cfg.paths.data
        >>> cfg["paths"]["data"]
    """

    def __init__(self, *args: Any, base_dir: Path | None = None, **kwargs: Any) -> None:
        """Initialize the configuration mapping.

        Args:
            *args: Positional arguments forwarded to ``dict``.
            base_dir: Directory against which relative paths are resolved.
            **kwargs: Keyword arguments forwarded to ``dict``.
        """
        super().__init__(*args, **kwargs)
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        for key, value in list(self.items()):
            if isinstance(value, dict):
                self[key] = Config(value, base_dir=self._base_dir)

    def __getattr__(self, key: str) -> Any:
        """Return the value for ``key`` via attribute access.

        Args:
            key: Attribute name, which must exist in the mapping.

        Returns:
            The stored value for ``key``.

        Raises:
            AttributeError: If the key is not present in the mapping.
        """
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"Config has no key {key!r}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        """Suppress setting of inherited internals by routing to the dict.

        Args:
            key: Attribute name.
            value: Value to store.
        """
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            if isinstance(value, dict) and not isinstance(value, Config):
                value = Config(value, base_dir=self._base_dir)
            self[key] = value

    def __dir__(self) -> list[str]:
        """List available keys for tab completion and introspection.

        Returns:
            Sorted list of mapping keys and class attributes.
        """
        return sorted(set(super().__dir__()) | set(self.keys()))

    def get_path(self, key: str) -> Path:
        """Resolve a path-valued config key relative to the YAML directory.

        Args:
            key: Dot-separated key path, e.g. ``"paths.data"``.

        Returns:
            Absolute :class:`pathlib.Path` for the requested key.

        Raises:
            KeyError: If the key does not exist.
            TypeError: If the stored value is not a string.
        """
        value = self.nested_get(key, default=_MISSING)
        if value is _MISSING:
            raise KeyError(f"Config key {key!r} does not exist")
        if not isinstance(value, str):
            raise TypeError(f"Config key {key!r} must be a string path, got {type(value)}")
        path = Path(value)
        return path if path.is_absolute() else (self._base_dir / path).resolve()

    def nested_get(self, key: str, default: Any = None) -> Any:
        """Traverse a dot-separated key path after simple key lookup.

        Args:
            key: Dot-separated key path, e.g. ``"paths.data"``.
            default: Value returned if any segment is missing.

        Returns:
            The nested value or ``default``.
        """
        if key in self:
            return self[key]
        result: Any = self
        for segment in key.split("."):
            if isinstance(result, dict) and segment in result:
                result = result[segment]
            elif isinstance(result, Config) and segment in result:
                result = result[segment]
            else:
                return default
        return result

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to plain nested dictionaries.

        Returns:
            The configuration as JSON-serializable plain dictionaries.
        """
        return {
            key: value.to_dict() if isinstance(value, Config) else value
            for key, value in self.items()
        }

    def save_yaml(self, path: str | Path) -> None:
        """Persist the configuration to a YAML file.

        Args:
            path: Destination YAML file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    def save_json(self, path: str | Path) -> None:
        """Persist the configuration to a JSON file.

        Args:
            path: Destination JSON file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            Compact string repr using plain dictionaries.
        """
        return f"Config({self.to_dict()!r})"

    def _iter_leaf_pairs(self) -> Iterator[tuple[str, Any]]:
        """Yield every leaf key and value with its full dot-separated path.

        Yields:
            Tuples of ``(dot_path, value)``.
        """
        for key, value in self.items():
            if isinstance(value, Config):
                for sub_key, sub_value in value._iter_leaf_pairs():
                    yield f"{key}.{sub_key}", sub_value
            else:
                yield key, value


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML configuration from ``path``.

    Args:
        path: Path to the YAML configuration file. Defaults to
            ``config/config.yaml`` relative to the repository root.

    Returns:
        The loaded :class:`Config` object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file cannot be parsed as YAML.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise yaml.YAMLError(f"Configuration root must be a mapping, got {type(raw)}")
    return Config(raw, base_dir=_resolve_base_dir(config_path))

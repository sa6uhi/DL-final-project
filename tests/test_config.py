"""Unit tests for :mod:`src.utils.config`."""

from __future__ import annotations

import pytest
import yaml

from src.utils.config import Config, load_config


@pytest.fixture()
def sample_yaml(tmp_path) -> str:
    """Write a sample configuration file and return its path."""
    content = {
        "seed": 42,
        "paths": {"data": "data/raw", "absolute": str(tmp_path / "abs_dir")},
        "nested": {"a": {"b": {"c": 7}}},
        "numbers": [1, 2, 3],
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(content), encoding="utf-8")
    return str(config_file)


def test_load_config_from_yaml(sample_yaml: str) -> None:
    """load_config parses a YAML file into a Config object."""
    cfg = load_config(sample_yaml)
    assert isinstance(cfg, Config)
    assert cfg["seed"] == 42
    assert cfg.nested.a.b.c == 7


def test_load_config_dot_access_equals_dict_access(sample_yaml: str) -> None:
    """Dot-notation and dict-notation return identical values."""
    cfg = load_config(sample_yaml)
    assert cfg.paths.data == cfg["paths"]["data"]
    assert cfg.nested.a.b.c == cfg["nested"]["a"]["b"]["c"]


def test_load_config_missing_file_raises(tmp_path) -> None:
    """A nonexistent configuration file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_malformed_yaml_raises(tmp_path) -> None:
    """Malformed YAML content raises yaml.YAMLError."""
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("seed: [unclosed\n  nested", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(bad_file)


def test_load_config_non_mapping_root_raises(tmp_path) -> None:
    """A non-mapping YAML root raises yaml.YAMLError."""
    bad_file = tmp_path / "list.yaml"
    bad_file.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(bad_file)


def test_nested_get_with_default(sample_yaml: str) -> None:
    """nested_get falls back gracefully for missing dotted paths."""
    cfg = load_config(sample_yaml)
    assert cfg.nested_get("nested.a.b.c") == 7
    assert cfg.nested_get("nested.a.b.d", default="fallback") == "fallback"
    assert cfg.nested_get("no.such.key", default=None) is None


def test_get_path_resolves_relative_against_yaml_dir(sample_yaml: str) -> None:
    """get_path resolves relative paths against the YAML directory."""
    cfg = load_config(sample_yaml)
    resolved = cfg.get_path("paths.data")
    path = pytest.importorskip("pathlib").Path(sample_yaml).parent / "data/raw"
    assert resolved == path.resolve()


def test_get_path_absolute_is_kept(sample_yaml: str) -> None:
    """get_path keeps absolute paths as-is."""
    cfg = load_config(sample_yaml)
    assert cfg.get_path("paths.absolute").is_absolute()


def test_get_path_from_config_dir_resolves_repo_root(tmp_path) -> None:
    """A config inside a ``config`` dir resolves paths against the repo root."""
    repo_root = tmp_path / "project"
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text("paths:\n  data: data/processed\n", encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.get_path("paths.data") == (repo_root / "data/processed").resolve()


def test_get_path_non_string_raises(sample_yaml: str) -> None:
    """get_path raises TypeError when the stored value is not a string."""
    cfg = load_config(sample_yaml)
    with pytest.raises(TypeError):
        cfg.get_path("seed")


def test_get_path_missing_key_raises(sample_yaml: str) -> None:
    """get_path raises KeyError for an absent dotted key."""
    cfg = load_config(sample_yaml)
    with pytest.raises(KeyError):
        cfg.get_path("does.not.exist")


def test_nested_values_are_config_instances(sample_yaml: str) -> None:
    """Nested dictionaries are wrapped as Config instances."""
    cfg = load_config(sample_yaml)
    assert isinstance(cfg.paths, Config)
    assert isinstance(cfg.nested.a, Config)


def test_to_dict_recursion(sample_yaml: str) -> None:
    """to_dict produces plain nested dictionaries."""
    cfg = load_config(sample_yaml)
    raw = cfg.to_dict()
    assert type(raw["nested"]["a"]) is dict


def test_save_yaml_roundtrip(sample_yaml: str, tmp_path) -> None:
    """Config survives a save/load YAML roundtrip."""
    cfg = load_config(sample_yaml)
    output = tmp_path / "out.yaml"
    cfg.save_yaml(output)
    reloaded = load_config(output)
    assert reloaded.to_dict() == cfg.to_dict()


def test_save_json_roundtrip(sample_yaml: str, tmp_path) -> None:
    """Config survives a save/load JSON roundtrip."""
    cfg = load_config(sample_yaml)
    output = tmp_path / "out.json"
    cfg.save_json(output)
    import json

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["nested"]["a"]["b"]["c"] == 7


def test_setattr_and_getattr(sample_yaml: str) -> None:
    """Assignment and lookup work through attribute access."""
    cfg = load_config(sample_yaml)
    cfg.new_section = {"x": 1}
    assert cfg.new_section.x == 1


def test_unknown_attribute_raises(sample_yaml: str) -> None:
    """Accessing an unknown attribute raises AttributeError."""
    cfg = load_config(sample_yaml)
    with pytest.raises(AttributeError):
        _ = cfg.definitely_not_a_key


def test_repr_is_evaluable_info(sample_yaml: str) -> None:
    """repr is a readable, short description."""
    cfg = load_config(sample_yaml)
    assert "Config" in repr(cfg)
    assert "seed" in repr(cfg)


def test_dir_lists_config_keys(sample_yaml: str) -> None:
    """dir() surfaces config keys for introspection."""
    cfg = load_config(sample_yaml)
    assert "paths" in dir(cfg)
    assert "seed" in dir(cfg)


def test_default_config_path_exists() -> None:
    """The repository's default config exists and validates as mapping."""
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 42


def test_iter_leaf_pairs_yields_dot_paths(sample_yaml: str) -> None:
    """_iter_leaf_pairs flattens nested keys into dot-separated paths."""
    cfg = load_config(sample_yaml)
    pairs = dict(cfg._iter_leaf_pairs())
    assert pairs["seed"] == 42
    assert pairs["nested.a.b.c"] == 7
    assert pairs["paths.data"] == "data/raw"

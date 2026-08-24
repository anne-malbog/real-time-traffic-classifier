"""Unit tests for scene config loading (src/scene_config.py)."""

import textwrap

from src.scene_config import load_scene_config


def test_loads_counting_lines(tmp_path):
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text(textwrap.dedent("""
        counting_lines:
          - name: main
            point1: [0, 100]
            point2: [200, 100]
            direction_a_label: north
            direction_b_label: south
        """))

    config = load_scene_config(config_path)

    assert len(config["counting_lines"]) == 1
    line = config["counting_lines"][0]
    assert line.name == "main"
    assert line.p1 == (0, 100)
    assert line.p2 == (200, 100)
    assert line.label_a == "north"
    assert line.label_b == "south"


def test_loads_density_thresholds_with_null_as_infinity(tmp_path):
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text(textwrap.dedent("""
        density:
          thresholds:
            - {max: 5, label: LOW}
            - {max: null, label: HIGH}
        """))

    config = load_scene_config(config_path)

    thresholds = config["density_thresholds"]
    assert thresholds[0].label == "LOW"
    assert thresholds[0].max_count == 5
    assert thresholds[1].label == "HIGH"
    assert thresholds[1].max_count == float("inf")


def test_loads_congestion_config_with_defaults_when_absent(tmp_path):
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text("counting_lines: []\n")

    config = load_scene_config(config_path)

    congestion = config["congestion_config"]
    assert congestion.density_trigger == "HIGH"
    assert congestion.slow_speed_threshold == 15.0
    assert congestion.min_slow_fraction == 0.5


def test_loads_congestion_config_overrides(tmp_path):
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text(textwrap.dedent("""
        congestion:
          density_trigger: MODERATE
          slow_speed_px_per_sec: 8.0
          min_slow_fraction: 0.75
        """))

    config = load_scene_config(config_path)

    congestion = config["congestion_config"]
    assert congestion.density_trigger == "MODERATE"
    assert congestion.slow_speed_threshold == 8.0
    assert congestion.min_slow_fraction == 0.75


def test_missing_optional_sections_produce_sensible_defaults(tmp_path):
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text("{}\n")

    config = load_scene_config(config_path)

    assert config["counting_lines"] == []
    assert config["density_thresholds"] is None  # caller falls back to DEFAULT_DENSITY_THRESHOLDS
    assert config["direction_reference"] == "screen"


def test_real_project_config_loads(tmp_path):
    """Sanity check against the actual configs/analytics.yaml shipped with the project."""
    config = load_scene_config("configs/analytics.yaml")

    assert len(config["counting_lines"]) >= 1
    assert config["density_thresholds"] is not None
    assert config["congestion_config"].density_trigger == "HIGH"

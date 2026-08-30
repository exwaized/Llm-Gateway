import pytest

from llm_gateway.config import ClassifierThresholds


def test_defaults_are_valid():
    thresholds = ClassifierThresholds()
    assert thresholds.cheap_max == 0.35
    assert thresholds.mid_max == 0.70


def test_rejects_cheap_max_above_mid_max():
    with pytest.raises(ValueError):
        ClassifierThresholds(cheap_max=0.8, mid_max=0.5)


def test_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        ClassifierThresholds(cheap_max=-0.1, mid_max=0.5)
    with pytest.raises(ValueError):
        ClassifierThresholds(cheap_max=0.1, mid_max=1.5)


def test_from_env_uses_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("GATEWAY_CHEAP_MAX", raising=False)
    monkeypatch.delenv("GATEWAY_MID_MAX", raising=False)

    thresholds = ClassifierThresholds.from_env()
    assert thresholds == ClassifierThresholds()


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("GATEWAY_CHEAP_MAX", "0.2")
    monkeypatch.setenv("GATEWAY_MID_MAX", "0.6")

    thresholds = ClassifierThresholds.from_env()
    assert thresholds.cheap_max == 0.2
    assert thresholds.mid_max == 0.6


def test_from_env_propagates_invalid_ordering(monkeypatch):
    monkeypatch.setenv("GATEWAY_CHEAP_MAX", "0.9")
    monkeypatch.setenv("GATEWAY_MID_MAX", "0.1")

    with pytest.raises(ValueError):
        ClassifierThresholds.from_env()

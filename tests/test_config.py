from src import config


def test_thresholds_are_ordered():
    assert 0 < config.LOW_RISK_MAX < config.HIGH_RISK_MIN < 1


def test_risk_band_boundaries():
    assert config.risk_band(0.0).startswith("Low")
    assert config.risk_band(config.LOW_RISK_MAX - 0.001).startswith("Low")
    assert config.risk_band(config.LOW_RISK_MAX).startswith("Medium")
    assert config.risk_band(config.HIGH_RISK_MIN - 0.001).startswith("Medium")
    assert config.risk_band(config.HIGH_RISK_MIN).startswith("High")
    assert config.risk_band(1.0).startswith("High")

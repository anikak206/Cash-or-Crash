import numpy as np

from src.cost_analysis import best_row, confusion_at, evaluate_threshold, sweep


def test_confusion_counts_add_up():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.3])
    tp, fp, fn, tn = confusion_at(y, p, 0.5)
    assert (tp, fp, fn, tn) == (1, 1, 1, 2)


def test_cost_formula():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.3])
    row = evaluate_threshold(y, p, 0.5, cost_default=100, cost_false_alarm=10)
    assert row["total_cost"] == 1 * 100 + 1 * 10


def test_higher_missed_cost_never_raises_best_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    low = best_row(sweep(y, p, 200, 100))["threshold"]
    high = best_row(sweep(y, p, 2000, 100))["threshold"]
    assert high <= low

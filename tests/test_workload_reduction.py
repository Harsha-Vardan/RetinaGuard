"""Tests for clinical workload-reduction / health economics."""

import numpy as np

from src.evaluation.workload_reduction import (
    compute_workload_reduction,
    workload_reduction_curve,
    health_economics_analysis,
)


def test_workload_basic():
    probs = np.array([[0.9, 0.05, 0.02, 0.02, 0.01],
                      [0.1, 0.1, 0.1, 0.1, 0.6],
                      [0.3, 0.3, 0.2, 0.1, 0.1]])
    y_true = np.array([0, 4, 1])
    m = compute_workload_reduction(y_true=y_true, y_prob=probs, alpha=0.1)
    assert 0.0 <= m["autonomous_clearance_rate"] <= 1.0
    assert m["n_samples"] == 3


import pytest


def test_workload_rates_sum():
    probs = np.array([[0.9, 0.05, 0.02, 0.02, 0.01],
                      [0.1, 0.1, 0.1, 0.1, 0.6]])
    m = compute_workload_reduction(y_prob=probs, alpha=0.1)
    assert m["autonomous_clearance_rate"] + m["referral_rate"] == pytest.approx(1.0)


def test_workload_curve_length():
    probs = np.random.rand(10, 5)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_true = np.random.randint(0, 5, size=10)
    curve = workload_reduction_curve(y_true=y_true, y_prob=probs, alphas=[0.05, 0.1, 0.2])
    assert len(curve) == 3
    for c in curve:
        assert "autonomous_clearance_rate" in c
        assert "alpha" in c


def test_health_economics():
    probs = np.array([[0.95, 0.02, 0.01, 0.01, 0.01]])
    m = health_economics_analysis(y_prob=probs, alpha=0.1, annual_patients=1000)
    assert "annual_hours_saved" in m
    assert m["annual_patients"] == 1000


def test_prediction_sets_input():
    sets = [[0], [0, 1, 2]]
    m = compute_workload_reduction(prediction_sets=sets)
    assert m["autonomous_clearance_rate"] == pytest.approx(0.5)
    assert m["referral_rate"] == pytest.approx(0.5)

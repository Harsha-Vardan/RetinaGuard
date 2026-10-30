"""Tests for asymmetric clinical cost matrix & ordinal risk loss."""

import numpy as np
import pytest
import torch

from src.training.clinical_cost import (
    clinical_cost_matrix,
    AsymmetricClinicalCostLoss,
    ordinal_risk_loss,
    compute_expected_cost,
)


def test_cost_matrix_spec_values():
    C = clinical_cost_matrix()
    assert C.shape == (5, 5)
    assert C[4, 0] == pytest.approx(10.0)
    assert C[0, 1] == pytest.approx(0.5)
    assert np.allclose(np.diag(C), 0.0)


def test_cost_matrix_false_negative_heavier():
    C = clinical_cost_matrix()
    # false negative (true 4 pred 0) >> false positive (true 0 pred 1)
    assert C[4, 0] > C[0, 1]
    # under-diagnosis costs positive
    assert C[3, 0] > 0


def test_asymmetric_loss_forward():
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 3, 4])
    loss_fn = AsymmetricClinicalCostLoss()
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_asymmetric_loss_components():
    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 3, 4])
    loss_fn = AsymmetricClinicalCostLoss()
    comp = loss_fn(logits, targets, return_components=True)
    assert "loss" in comp and "clinical_cost_loss" in comp
    assert comp["loss"].item() >= 0


def test_ordinal_risk_loss_zero_when_perfect():
    logits = torch.zeros(3, 5)
    logits[:, 2] = 10.0
    targets = torch.tensor([2, 2, 2])
    loss = ordinal_risk_loss(logits, targets)
    assert loss.item() < 1e-3


def test_compute_expected_cost():
    y_true = np.array([4, 0])
    y_pred = np.array([0, 1])
    cost = compute_expected_cost(y_true, y_pred)
    C = clinical_cost_matrix()
    expected = (C[4, 0] + C[0, 1]) / 2.0
    assert cost == pytest.approx(expected)


def test_custom_matrix():
    M = np.ones((3, 3)) * 2.0
    np.fill_diagonal(M, 0.0)
    C = clinical_cost_matrix(num_classes=3, matrix=M)
    assert np.allclose(C, M)

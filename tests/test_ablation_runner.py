"""RetinaGuard - Tests for Systematic Ablation Experiment Runner."""

from pathlib import Path
import pandas as pd
import pytest
import torch

from src.evaluation.run_ablations import AblationRunner


@pytest.fixture
def runner():
    """Returns an AblationRunner configured in smoke_test mode."""
    return AblationRunner(smoke_test=True)


def test_ablation_runner_initialization(runner):
    """Verify runner initializes directories and transforms."""
    assert runner.smoke_test is True
    assert runner.device is not None
    assert runner.reports_dir.exists()
    assert runner.figures_dir.exists()


def test_ablation_runner_load_test_manifests(runner):
    """Verify manifest loading yields stratified test subsets."""
    idrid_df, aptos_df, lesion_pairs = runner.load_test_manifests()
    assert isinstance(idrid_df, pd.DataFrame)
    assert len(idrid_df) > 0
    assert "image_path" in idrid_df.columns
    assert "dr_grade" in idrid_df.columns

    assert isinstance(aptos_df, pd.DataFrame)
    assert isinstance(lesion_pairs, list)


def test_ablation_runner_build_models(runner):
    """Verify all 4 ablation variants are properly instantiated."""
    models = runner.build_ablation_models()
    assert len(models) == 4
    expected_names = [
        "Baseline (ResNet18)",
        "ResNet18 + Lesion Attention",
        "ResNet18 + Causal Disentangler",
        "RetinaGuard Unified",
    ]
    for name in expected_names:
        assert name in models
        info = models[name]
        assert "model" in info
        assert "type" in info
        assert "loss_terms" in info
        assert isinstance(info["model"], torch.nn.Module)


def test_ablation_runner_evaluate_attribution_masks(runner):
    """Verify attribution mask evaluation produces valid IoU and Dice scores in [0, 1]."""
    models = runner.build_ablation_models()
    rg_info = models["RetinaGuard Unified"]

    iou, dice = runner.evaluate_attribution_masks(rg_info, lesion_pairs=[])
    assert 0.0 <= iou <= 1.0
    assert 0.0 <= dice <= 1.0
    assert dice >= iou  # Mathematically, Dice >= IoU


def test_ablation_runner_write_markdown_report(runner, tmp_path):
    """Verify markdown report generation without tabulate dependency."""
    df = pd.DataFrame([
        {
            "Architecture Configuration": "Baseline (ResNet18)",
            "Objective Formulation": "L_CE",
            "In-Domain QWK (IDRiD)": 0.82,
            "Domain Gap (Δ Acc)": 0.23,
        },
        {
            "Architecture Configuration": "RetinaGuard Unified",
            "Objective Formulation": "L_total",
            "In-Domain QWK (IDRiD)": 0.89,
            "Domain Gap (Δ Acc)": 0.06,
        },
    ])
    out_file = tmp_path / "test_ablation.md"
    runner._write_markdown_report(df, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# RetinaGuard: Component Ablation Study" in content
    assert "| Architecture Configuration | Objective Formulation |" in content
    assert "RetinaGuard Unified" in content

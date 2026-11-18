"""Unit tests for Phase 2 depth engine: registry, refinement, uncertainty proxy, and hand-computed depth metrics."""

import math
import numpy as np
import pytest

from app.core.exceptions import DepthEstimationError
from app.evaluation.depth_metrics import (
    align_depth_least_squares,
    compute_depth_metrics,
)
from app.vision.depth_estimation import (
    create_depth_model,
    list_available_depth_models,
)
from app.vision.depth_refinement import (
    detect_depth_discontinuities,
    refine_depth_map,
)
from app.vision.uncertainty import compute_uncertainty_proxy


def test_depth_model_registry_selection() -> None:
    """Verify at least 3 models are registered and swappable by config name."""
    models = list_available_depth_models()
    names = {m["model_name"] for m in models}
    assert "depth_anything_v2_small" in names
    assert "dpt_hybrid_midas" in names
    assert "depth_anything_v2_metric_indoor_small" in names

    m1 = create_depth_model("depth_anything_v2_small", device="cpu")
    assert m1.is_metric is False
    m2 = create_depth_model("depth_anything_v2_metric_indoor_small", device="cpu")
    assert m2.is_metric is True

    with pytest.raises(DepthEstimationError):
        create_depth_model("non_existent_model_xyz")


def test_hand_computed_depth_metrics_unaligned() -> None:
    """Hand-computed check for Abs Rel, Sq Rel, RMSE, RMSE log, and delta thresholds:
    Let GT = [[2.0, 4.0], [5.0, 8.0]]
    Let Pred = [[2.2, 3.6], [5.0, 10.4]]
    Errors (p - g):
      pixel 1: 2.2 - 2.0 = +0.2; |e|/g = 0.2/2.0 = 0.10; e^2/g = 0.04/2.0 = 0.02; e^2 = 0.04; ratio = 1.10 (<1.25)
      pixel 2: 3.6 - 4.0 = -0.4; |e|/g = 0.4/4.0 = 0.10; e^2/g = 0.16/4.0 = 0.04; e^2 = 0.16; ratio = 4.0/3.6 = 1.1111 (<1.25)
      pixel 3: 5.0 - 5.0 =  0.0; |e|/g = 0.0;            e^2/g = 0.0;           e^2 = 0.00; ratio = 1.00 (<1.25)
      pixel 4: 10.4 - 8.0 = +2.4; |e|/g = 2.4/8.0 = 0.30; e^2/g = 5.76/8.0 = 0.72; e^2 = 5.76; ratio = 10.4/8.0 = 1.30 (>=1.25, <1.5625)
    Expected:
      Abs Rel = (0.10 + 0.10 + 0.00 + 0.30) / 4 = 0.125
      Sq Rel  = (0.02 + 0.04 + 0.00 + 0.72) / 4 = 0.195
      RMSE    = sqrt((0.04 + 0.16 + 0.00 + 5.76) / 4) = sqrt(5.96 / 4) = sqrt(1.49) = 1.2206555615733703
      delta_1 (<1.25)   = 3 / 4 = 0.75
      delta_2 (<1.5625) = 4 / 4 = 1.00
      delta_3 (<1.9531) = 4 / 4 = 1.00
    """
    gt = np.array([[2.0, 4.0], [5.0, 8.0]], dtype=np.float32)
    pred = np.array([[2.2, 3.6], [5.0, 10.4]], dtype=np.float32)

    m = compute_depth_metrics(
        pred_depth=pred,
        gt_depth=gt,
        alignment_mode="none",
        min_depth=0.1,
        max_depth=20.0,
    )

    expected_rmse_log = math.sqrt(
        (
            math.log(2.2 / 2.0) ** 2
            + math.log(3.6 / 4.0) ** 2
            + math.log(5.0 / 5.0) ** 2
            + math.log(10.4 / 8.0) ** 2
        )
        / 4.0
    )

    assert m.abs_rel == pytest.approx(0.125, abs=1e-6)
    assert m.sq_rel == pytest.approx(0.195, abs=1e-6)
    assert m.rmse == pytest.approx(math.sqrt(1.49), abs=1e-6)
    assert m.rmse_log == pytest.approx(expected_rmse_log, abs=1e-6)
    assert m.delta_1 == pytest.approx(0.75, abs=1e-6)
    assert m.delta_2 == pytest.approx(1.0, abs=1e-6)
    assert m.delta_3 == pytest.approx(1.0, abs=1e-6)


def test_least_squares_scale_shift_recovery() -> None:
    """Verify exact recovery of scale s=2.5 and shift t=0.75 from affine-transformed depth."""
    raw = np.linspace(1.0, 5.0, 64, dtype=np.float32).reshape(8, 8)
    gt = 2.5 * raw + 0.75
    valid = np.ones_like(gt, dtype=bool)

    aligned, s, t = align_depth_least_squares(raw, gt, valid, min_depth=0.1, max_depth=20.0)
    assert s == pytest.approx(2.5, abs=1e-5)
    assert t == pytest.approx(0.75, abs=1e-5)
    np.testing.assert_allclose(aligned, gt, atol=1e-5)


def test_edge_preserving_refinement_reduces_noise_and_keeps_step_edge() -> None:
    """Verify guided and bilateral refinement reduce flat-region noise while preserving step jump."""
    rng = np.random.default_rng(42)
    h, w = 64, 64
    clean_depth = np.full((h, w), 2.0, dtype=np.float32)
    clean_depth[:, 32:] = 6.0  # Sharp step edge at u=32

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :32] = 60
    rgb[:, 32:] = 220

    noisy_depth = clean_depth + rng.normal(0.0, 0.05, size=(h, w)).astype(np.float32)
    rmse_noisy = float(np.sqrt(np.mean((noisy_depth - clean_depth) ** 2)))

    res_guided = refine_depth_map(noisy_depth, rgb, method="guided", guided_radius=3, guided_eps=1e-3)
    res_bilat = refine_depth_map(noisy_depth, rgb, method="bilateral", bilateral_d=5)

    rmse_guided = float(np.sqrt(np.mean((res_guided.refined_depth - clean_depth) ** 2)))
    rmse_bilat = float(np.sqrt(np.mean((res_bilat.refined_depth - clean_depth) ** 2)))

    assert rmse_guided < rmse_noisy
    assert rmse_bilat < rmse_noisy

    # Step jump across column 31 -> 32 must still be > 3.5 units (not blurred into a ramp)
    step_jump = float(np.mean(res_guided.refined_depth[:, 32] - res_guided.refined_depth[:, 31]))
    assert step_jump > 3.5

    disc_mask = detect_depth_discontinuities(clean_depth, relative_jump_threshold=0.1)
    assert np.all(disc_mask[:, 31]) and np.all(disc_mask[:, 32])
    assert not np.any(disc_mask[:, 10])


def test_uncertainty_proxy_peaks_at_discontinuities_and_inconsistencies() -> None:
    """Verify uncertainty proxy map is in [0, 1] and peaks at depth jumps and flip mismatches."""
    depth = np.full((32, 32), 2.0, dtype=np.float32)
    depth[:, 16:] = 8.0

    flipped = depth.copy()
    flipped[0:8, 0:8] = 5.0  # Synthetic flip inconsistency in top-left corner

    res = compute_uncertainty_proxy(
        depth_map=depth,
        flipped_depth_map=flipped,
        uncertainty_threshold=0.5,
    )
    assert res.uncertainty_proxy_map.shape == (32, 32)
    assert np.all(res.uncertainty_proxy_map >= 0.0)
    assert np.all(res.uncertainty_proxy_map <= 1.0)
    assert "depth_gradient_discontinuity" in res.components_used
    assert "horizontal_flip_consistency" in res.components_used

    # Boundary columns (15, 16) must have much higher uncertainty proxy than uniform interior (col 24)
    assert float(np.mean(res.uncertainty_proxy_map[:, 16])) > float(
        np.mean(res.uncertainty_proxy_map[:, 24])
    ) + 0.25

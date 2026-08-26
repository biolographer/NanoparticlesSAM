"""
Unit tests for model_comparison.py - the module that scores base-vs-fine-tuned
SAM2 detection quality against the held-out `data/validation data/` set's
manually-drawn ground-truth circles.

Matching/metric functions are exercised with synthetic filtered_df + ground
truth data (same StubMaskGenerator-style approach as test_sphere_segmentation.py)
so most of this runs without SAM2 or a GPU. The Wilcoxon comparison is checked
against scipy.stats.wilcoxon directly, exactly as test_particle_stats.py checks
the (unrelated) Welch's t-test against scipy - this is a deliberately separate
test file/statistic, per the project's requirement that the new model-comparison
significance test not be coupled to particle_stats.py.

A couple of tests also run against the real fixture
data/validation data/A2-24_1.tif (8 annotated circles, one of them clipped by
the image border) to confirm the real ground-truth-parsing wiring end to end.
"""
import os

import cv2
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from model_comparison import (
    compare_models_wilcoxon,
    evaluate_image,
    evaluate_model_on_validation_set,
    filter_ground_truth_near_border,
    is_near_border,
    load_validation_item,
    match_detections_to_ground_truth,
)

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "validation data")
FIXTURE_TIF = os.path.join(FIXTURE_DIR, "A2-24_1.tif")
FIXTURE_TIF_NO_GT = os.path.join(FIXTURE_DIR, "A2h_01.tif")


def make_filtered_df(entries):
    """entries: list of (x, y, feret_diameter)."""
    return pd.DataFrame({
        "point_coords_tuple": [(x, y) for x, y, _ in entries],
        "feret_diameter": [d for _, _, d in entries],
    })


def test_match_detections_to_ground_truth_pairs_close_points_and_ignores_far_ones():
    ground_truth = [((100, 100), 20), ((500, 500), 20)]
    filtered_df = make_filtered_df([
        (102, 98, 40),    # close to the first gt circle -> should match
        (900, 900, 40),   # far from every gt circle -> should not match
    ])

    matches = match_detections_to_ground_truth(filtered_df, ground_truth)

    assert len(matches) == 1
    assert matches[0]["gt_center"] == (100, 100)
    assert matches[0]["det_index"] == 0


def test_match_detections_to_ground_truth_returns_empty_when_nothing_to_match():
    assert match_detections_to_ground_truth(None, [((0, 0), 10)]) == []
    assert match_detections_to_ground_truth(make_filtered_df([(0, 0, 10)]), []) == []


def test_is_near_border_flags_circles_that_touch_any_edge():
    height, width = 960, 1280
    assert is_near_border((10, 500), 20, height, width)      # too close to left edge
    assert is_near_border((500, 10), 20, height, width)       # too close to top edge
    assert is_near_border((1270, 500), 20, height, width)     # too close to right edge
    assert is_near_border((500, 950), 20, height, width)      # too close to bottom edge
    assert not is_near_border((500, 500), 20, height, width)  # comfortably interior


def test_filter_ground_truth_near_border_drops_only_border_circles():
    ground_truth = [((500, 500), 20), ((5, 500), 20)]
    filtered = filter_ground_truth_near_border(ground_truth, height=960, width=1280)
    assert filtered == [((500, 500), 20)]


def test_evaluate_image_computes_precision_recall_f1_and_radius_error():
    ground_truth = [((100, 100), 20), ((500, 500), 25)]
    # detection 1 matches gt 1 with a +5 radius overestimate (feret_diameter=50 -> radius 25)
    # detection 2 is a spurious extra detection with no matching ground truth
    filtered_df = make_filtered_df([(102, 98, 50), (900, 900, 40)])

    metrics = evaluate_image(filtered_df, ground_truth, image_shape=(960, 1280, 3), filter_border=False)

    assert metrics["n_gt_total"] == 2
    assert metrics["n_gt"] == 2
    assert metrics["n_detected"] == 2
    assert metrics["n_matched"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["mean_abs_radius_error"] == pytest.approx(5.0)
    assert metrics["mean_signed_radius_error"] == pytest.approx(5.0)


def test_evaluate_image_returns_nan_metrics_when_nothing_detected():
    ground_truth = [((100, 100), 20)]
    metrics = evaluate_image(None, ground_truth, image_shape=(960, 1280, 3), filter_border=False)

    assert metrics["n_detected"] == 0
    assert metrics["n_matched"] == 0
    assert np.isnan(metrics["precision"])
    assert metrics["recall"] == pytest.approx(0.0)
    assert np.isnan(metrics["f1"])


def test_evaluate_image_border_filtering_changes_recall_and_gt_counts():
    # One interior circle (detected) and one border-clipped circle (never
    # detected, since sphere_segmentation's own border_cutoff would drop a
    # detection there too) - filtering the border circle out of the ground
    # truth is what keeps recall from being unfairly penalized.
    interior = ((500, 500), 30)
    border_clipped = ((5, 500), 30)
    ground_truth = [interior, border_clipped]
    filtered_df = make_filtered_df([(500, 500, 60)])
    image_shape = (960, 1280, 3)

    with_border_filter = evaluate_image(filtered_df, ground_truth, image_shape, filter_border=True)
    assert with_border_filter["n_gt_total"] == 2
    assert with_border_filter["n_gt"] == 1
    assert with_border_filter["n_detected"] == 1
    assert with_border_filter["n_matched"] == 1
    assert with_border_filter["recall"] == pytest.approx(1.0)

    without_border_filter = evaluate_image(filtered_df, ground_truth, image_shape, filter_border=False)
    assert without_border_filter["n_gt_total"] == 2
    assert without_border_filter["n_gt"] == 2
    assert without_border_filter["n_detected"] == 1
    assert without_border_filter["n_matched"] == 1
    assert without_border_filter["recall"] == pytest.approx(0.5)


def test_compare_models_wilcoxon_matches_scipy_reference():
    rng = np.random.default_rng(5)
    img_names = [f"img_{i}.tif" for i in range(12)]
    base_vals = rng.normal(loc=10, scale=2, size=12)
    finetuned_vals = base_vals - rng.normal(loc=3, scale=1, size=12)  # consistently smaller error

    base_df = pd.DataFrame({"img_name": img_names, "mean_abs_radius_error": base_vals})
    finetuned_df = pd.DataFrame({"img_name": img_names, "mean_abs_radius_error": finetuned_vals})

    result = compare_models_wilcoxon(base_df, finetuned_df)
    scipy_result = stats.wilcoxon(finetuned_vals, base_vals)

    assert result["metric"] == "mean_abs_radius_error"
    assert result["n_pairs"] == 12
    assert result["statistic"] == pytest.approx(scipy_result.statistic)
    assert result["p_value"] == pytest.approx(scipy_result.pvalue)
    assert result["median_finetuned"] < result["median_base"]


def test_compare_models_wilcoxon_only_pairs_images_present_in_both():
    base_df = pd.DataFrame({"img_name": ["a.tif", "b.tif", "c.tif"], "f1": [0.5, 0.6, 0.7]})
    finetuned_df = pd.DataFrame({"img_name": ["a.tif", "b.tif"], "f1": [0.8, 0.9]})

    result = compare_models_wilcoxon(base_df, finetuned_df, metric="f1")

    assert result["n_pairs"] == 2


def test_load_validation_item_parses_real_ground_truth_circles():
    image, ground_truth = load_validation_item(FIXTURE_TIF)

    assert image.shape == (960, 1280, 3)
    assert len(ground_truth) == 8
    # One of the 8 annotated circles is clipped by the left border.
    assert sum(is_near_border(c, r, 960, 1280) for c, r in ground_truth) == 1


def test_load_validation_item_returns_none_without_shapepoints_metadata():
    assert load_validation_item(FIXTURE_TIF_NO_GT) is None


def test_evaluate_model_on_validation_set_end_to_end_against_real_fixture(tmp_path):
    import shutil
    shutil.copy(FIXTURE_TIF, tmp_path / "A2-24_1.tif")

    image, ground_truth = load_validation_item(str(tmp_path / "A2-24_1.tif"))
    interior_gt = [g for g in ground_truth if not is_near_border(g[0], g[1], *image.shape[:2])]
    detected_radii = [88, 90, 92, 94, 96, 89, 93]

    masks = []
    for (center, _), radius in zip(interior_gt, detected_radii):
        seg = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(seg, (int(center[0]), int(center[1])), radius, 1, thickness=-1)
        seg = seg.astype(bool)
        masks.append({"segmentation": seg, "area": int(seg.sum()), "point_coords": [[int(center[0]), int(center[1])]]})

    class StubMaskGenerator:
        def generate(self, img):
            return masks

    df = evaluate_model_on_validation_set(
        tmp_path, StubMaskGenerator(),
        sphere_segmentation_kwargs={
            "border_cutoff": True, "max_feret_filter": False,
            "min_feret_filter": False, "hough_circles": False,
        },
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["img_name"] == "A2-24_1.tif"
    assert row["n_gt_total"] == 8
    assert row["n_gt"] == 7
    assert row["n_detected"] == 5
    assert row["n_matched"] == 5
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(5 / 7)
    assert row["mean_abs_radius_error"] < 5.0

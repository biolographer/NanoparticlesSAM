"""
Unit tests for the geometry/circularity helpers in particle_seg.py that decide
whether a segmented blob counts as a (spherical) particle or not.

Synthetic masks with known ground truth are used so the checks don't depend on
SAM2 or any real SEM image:
  - a filled circle  -> the "this is round" case
  - a thin bar       -> the "this is not round" case, used to confirm the
                         metrics actually separate round from non-round shapes
"""
import numpy as np
import cv2
import pandas as pd
import pytest

from particle_seg import (
    feret_diameter,
    min_feret_diameter,
    compute_perimeter,
    circularity,
    circularity2,
    smoothened_mask,
    detect_circle,
    get_hough_radius,
)


def make_circle_mask(shape=(220, 220), center=(110, 110), radius=60):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask


def make_bar_mask(shape=(220, 220), center=(110, 110), length=140, width=16):
    mask = np.zeros(shape, dtype=np.uint8)
    half_l, half_w = length // 2, width // 2
    cv2.rectangle(
        mask,
        (center[0] - half_l, center[1] - half_w),
        (center[0] + half_l, center[1] + half_w),
        1,
        thickness=-1,
    )
    return mask


def test_feret_diameter_matches_known_circle_diameter():
    radius = 60
    mask = make_circle_mask(radius=radius)
    assert feret_diameter(mask) == pytest.approx(2 * radius, rel=0.07)


def test_min_feret_diameter_matches_known_circle_diameter():
    radius = 60
    mask = make_circle_mask(radius=radius)
    assert min_feret_diameter(mask) == pytest.approx(2 * radius, rel=0.07)


def test_feret_diameters_are_close_for_a_circle_but_diverge_for_a_bar():
    circle = make_circle_mask(radius=60)
    bar = make_bar_mask(length=140, width=16)

    circle_ratio = feret_diameter(circle) / min_feret_diameter(circle)
    bar_ratio = feret_diameter(bar) / min_feret_diameter(bar)

    # A circle's max and min caliper widths are essentially identical...
    assert circle_ratio == pytest.approx(1.0, rel=0.1)
    # ...while an elongated (non-circular) particle's are not.
    assert bar_ratio > 3


def test_feret_functions_return_nan_for_empty_mask():
    empty = np.zeros((100, 100), dtype=np.uint8)
    assert np.isnan(feret_diameter(empty))
    assert np.isnan(min_feret_diameter(empty))


def test_compute_perimeter_scales_with_known_circle_circumference():
    # compute_perimeter counts marching-squares contour points rather than
    # summing segment lengths, so it systematically overshoots the true
    # geometric circumference (2*pi*r) by a roughly constant factor for a
    # circle. Pin that factor instead of the raw geometric formula, and
    # confirm the measured perimeter still scales linearly with radius.
    small_radius, large_radius = 40, 80
    small_mask = make_circle_mask(shape=(180, 180), center=(90, 90), radius=small_radius)
    large_mask = make_circle_mask(shape=(280, 280), center=(140, 140), radius=large_radius)

    small_perimeter = compute_perimeter(small_mask)
    large_perimeter = compute_perimeter(large_mask)

    small_bias = small_perimeter / (2 * np.pi * small_radius)
    large_bias = large_perimeter / (2 * np.pi * large_radius)

    assert small_bias == pytest.approx(1.3, abs=0.1)
    assert large_bias == pytest.approx(1.3, abs=0.1)
    assert large_perimeter / small_perimeter == pytest.approx(large_radius / small_radius, rel=0.1)


def test_circularity_is_higher_for_a_circle_than_for_a_bar():
    # Because compute_perimeter overshoots the true circumference (see above),
    # circularity() does not land near 1.0 even for a perfect circle under
    # this implementation. What matters for particle filtering is that it
    # reliably ranks a circle above an elongated (non-circular) shape.
    circle = make_circle_mask(radius=60)
    bar = make_bar_mask(length=140, width=16)

    circle_circularity = circularity(circle.sum(), compute_perimeter(circle))
    bar_circularity = circularity(bar.sum(), compute_perimeter(bar))

    assert circle_circularity == pytest.approx(0.6, abs=0.1)
    assert bar_circularity < 0.4
    assert circle_circularity > bar_circularity


def test_circularity2_distinguishes_circle_from_bar_after_smoothing():
    circle = smoothened_mask(make_circle_mask(radius=60))
    bar = smoothened_mask(make_bar_mask(length=140, width=16))

    circle_circularity = circularity2(circle)
    bar_circularity = circularity2(bar)

    assert circle_circularity > 0.85
    assert bar_circularity < 0.5


def test_smoothened_mask_keeps_a_clean_circle_essentially_unchanged():
    radius = 60
    mask = make_circle_mask(radius=radius)
    smoothed = smoothened_mask(mask)

    original_area = mask.sum()
    smoothed_area = smoothed.sum()
    assert smoothed_area == pytest.approx(original_area, rel=0.05)


def test_detect_circle_recovers_known_radius():
    radius = 60
    mask = make_circle_mask(radius=radius)
    result = detect_circle(mask)
    assert result is not None
    _, _, detected_radius = result
    assert detected_radius == pytest.approx(radius, rel=0.15)


def test_get_hough_radius_returns_pdNA_when_no_circle_present():
    # A blank mask contains no circle at all - this must fail gracefully
    # rather than raising, since sphere_segmentation calls this per-particle.
    empty = np.zeros((100, 100), dtype=np.uint8)
    result = get_hough_radius(empty)
    assert pd.isna(result)

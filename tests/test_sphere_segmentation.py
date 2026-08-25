"""
Unit tests for sphere_segmentation() in particle_seg.py - the function that
turns raw mask-generator output into the filtered particle table used
downstream for size analysis.

A stub mask generator stands in for SAM2AutomaticMaskGenerator so this can
run without SAM2 or a GPU. It returns a mix of circular AND non-circular
(bar-shaped) synthetic masks, since distinguishing those two is exactly what
sphere_segmentation's circularity filter is for.

Several circles with different areas are used (not just one) because
sphere_segmentation applies an IQR filter on particle area: with too few
data points the 5th/95th percentile can land on the extreme values
themselves, degenerately filtering out the only surviving particle(s).
"""
import numpy as np
import cv2
import pytest

from particle_seg import sphere_segmentation


class StubMaskGenerator:
    """Stands in for SAM2AutomaticMaskGenerator: returns pre-built masks."""

    def __init__(self, masks):
        self._masks = masks

    def generate(self, img):
        return self._masks


def make_circle_mask(shape, center, radius):
    tmp = np.zeros(shape, dtype=np.uint8)
    cv2.circle(tmp, center, radius, 1, thickness=-1)
    return tmp.astype(bool)


def make_bar_mask(shape, center, length, width):
    tmp = np.zeros(shape, dtype=np.uint8)
    half_l, half_w = length // 2, width // 2
    cv2.rectangle(
        tmp,
        (center[0] - half_l, center[1] - half_w),
        (center[0] + half_l, center[1] + half_w),
        1,
        thickness=-1,
    )
    return tmp.astype(bool)


def build_masks(shape, circle_specs, bar_specs):
    masks = []
    for center, radius in circle_specs:
        seg = make_circle_mask(shape, center, radius)
        masks.append({"segmentation": seg, "area": int(seg.sum()), "point_coords": [[center[0], center[1]]]})
    for center, length, width in bar_specs:
        seg = make_bar_mask(shape, center, length, width)
        masks.append({"segmentation": seg, "area": int(seg.sum()), "point_coords": [[center[0], center[1]]]})
    return masks


SHAPE = (1000, 1000)
CIRCLE_SPECS = [
    ((150, 150), 40),
    ((450, 150), 50),
    ((750, 150), 60),
    ((150, 450), 70),
    ((450, 450), 80),
]
BAR_SPECS = [
    ((750, 450), 200, 20),
    ((150, 750), 160, 30),
]


def run_sphere_segmentation():
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    masks = build_masks(SHAPE, CIRCLE_SPECS, BAR_SPECS)
    mask_generator = StubMaskGenerator(masks)
    return sphere_segmentation(
        img,
        mask_generator,
        border_cutoff=True,
        max_feret_filter=False,
        min_feret_filter=False,
        hough_circles=False,
    )


def test_sphere_segmentation_drops_all_non_circular_masks():
    _, _, filtered_df = run_sphere_segmentation()

    bar_centers = {center for center, _, _ in BAR_SPECS}
    surviving_centers = set(filtered_df["point_coords_tuple"])

    assert bar_centers.isdisjoint(surviving_centers)


def test_sphere_segmentation_keeps_only_mid_range_circular_areas():
    # The two extreme-area circles (radius 40 and 80) fall outside the 5th/95th
    # percentile of the surviving circles' areas and get trimmed by the IQR
    # area filter; the three mid-range circles (radius 50, 60, 70) survive.
    _, _, filtered_df = run_sphere_segmentation()

    circle_centers_by_radius = {radius: center for center, radius in CIRCLE_SPECS}
    surviving_centers = set(filtered_df["point_coords_tuple"])

    assert surviving_centers == {
        circle_centers_by_radius[50],
        circle_centers_by_radius[60],
        circle_centers_by_radius[70],
    }


def test_sphere_segmentation_reports_high_circularity_for_surviving_particles():
    _, _, filtered_df = run_sphere_segmentation()

    assert len(filtered_df) > 0
    assert (filtered_df["circularity"] > 0.75).all()


def test_sphere_segmentation_returns_none_when_generator_finds_nothing():
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    mask_generator = StubMaskGenerator([])

    comb_mask, combined_array, filtered_df = sphere_segmentation(
        img, mask_generator, border_cutoff=True, max_feret_filter=False,
        min_feret_filter=False, hough_circles=False,
    )

    assert comb_mask is None
    assert combined_array is None
    assert filtered_df is None

"""
Compares a base SAM2 checkpoint against a fine-tuned checkpoint on a held-out
validation set, using the real automatic-detection inference path
(SAM2AutomaticMaskGenerator + sphere_segmentation, as in sam2_predictor.ipynb)
rather than the point-prompted path used during training.

Ground truth comes from the manually-drawn circle annotations
(dataset.get_circle_metadata / get_circle_from_points) already present on the
validation tifs - not from JEOL pixel-calibration metadata, which those tifs
don't carry.

The significance test here (Wilcoxon signed-rank, paired per image) is
intentionally standalone: it does not use or modify particle_stats.py, whose
Welch's t-test machinery is scoped to comparing physical experiment
conditions, not models.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

from dataset import get_circle_metadata, get_circle_from_points
from particle_seg import sphere_segmentation


def load_validation_item(tif_path, crop_banner=True, crop_height=960, crop_width=1280):
    """Loads one validation tif plus its ground-truth circles.

    Returns (image, ground_truth) where ground_truth is a list of
    (center_xy, radius) tuples in pixel space, or None if the tif carries no
    ShapePoints annotation.
    """
    metadata = get_circle_metadata(tif_path)
    if metadata is None:
        return None

    image = cv2.imread(str(tif_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_banner:
        image = image[:crop_height, :crop_width, ...]

    ground_truth = [get_circle_from_points(*triplet) for triplet in metadata]
    return image, ground_truth


def is_near_border(center, radius, height, width):
    """Same rule as particle_seg.remove_border_particles, applied to a plain
    (center, radius) pair instead of a filtered_df row."""
    x, y = center
    return (x - radius < 0) or (y - radius < 0) or (x + radius > width) or (y + radius > height)


def filter_ground_truth_near_border(ground_truth, height, width):
    return [(center, radius) for center, radius in ground_truth
            if not is_near_border(center, radius, height, width)]


def match_detections_to_ground_truth(filtered_df, ground_truth, max_distance_factor=1.0):
    """Optimal one-to-one matching (Hungarian algorithm) between ground-truth
    circles and detected particles, by center-to-center distance.

    A pair is only kept if its distance is within max_distance_factor *
    gt_radius - matches beyond that are treated as unrelated detections/misses.
    """
    if filtered_df is None or len(filtered_df) == 0 or len(ground_truth) == 0:
        return []

    det_centers = np.array(list(filtered_df["point_coords_tuple"]))
    gt_centers = np.array([center for center, _ in ground_truth])
    gt_radii = np.array([radius for _, radius in ground_truth])

    distances = cdist(gt_centers, det_centers)
    gt_indices, det_indices = linear_sum_assignment(distances)

    matches = []
    for gt_i, det_i in zip(gt_indices, det_indices):
        if distances[gt_i, det_i] <= gt_radii[gt_i] * max_distance_factor:
            matches.append({
                "gt_center": tuple(gt_centers[gt_i]),
                "gt_radius": gt_radii[gt_i],
                "det_index": filtered_df.index[det_i],
            })
    return matches


def evaluate_image(filtered_df, ground_truth, image_shape, radius_column="feret_diameter",
                    nanometer_per_pixel=None, filter_border=True):
    """Per-image detected-vs-annotated comparison.

    n_gt_total is every annotated circle; n_gt is the count actually used for
    precision/recall (border-clipped annotations excluded when filter_border
    is True, since sphere_segmentation's own border_cutoff would drop a
    detection there regardless of model quality).
    """
    height, width = image_shape[:2]
    n_gt_total = len(ground_truth)

    gt_for_eval = filter_ground_truth_near_border(ground_truth, height, width) if filter_border else ground_truth
    n_gt = len(gt_for_eval)
    n_detected = 0 if filtered_df is None else len(filtered_df)

    matches = match_detections_to_ground_truth(filtered_df, gt_for_eval)
    n_matched = len(matches)

    precision = n_matched / n_detected if n_detected else np.nan
    recall = n_matched / n_gt if n_gt else np.nan
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    scale = nanometer_per_pixel if nanometer_per_pixel else 1
    radius_errors = []
    for match in matches:
        det_radius = filtered_df.loc[match["det_index"], radius_column] / 2.0
        radius_errors.append((det_radius - match["gt_radius"]) * scale)

    return {
        "n_gt_total": n_gt_total,
        "n_gt": n_gt,
        "n_detected": n_detected,
        "n_matched": n_matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_radius_error": np.mean(np.abs(radius_errors)) if radius_errors else np.nan,
        "median_abs_radius_error": np.median(np.abs(radius_errors)) if radius_errors else np.nan,
        "mean_signed_radius_error": np.mean(radius_errors) if radius_errors else np.nan,
    }


def evaluate_model_on_validation_set(validation_dir, mask_generator, nanometer_per_pixel=None,
                                      crop_banner=True, filter_border=True,
                                      sphere_segmentation_kwargs=None):
    """Runs sphere_segmentation over every annotated tif in validation_dir and
    returns one row of evaluate_image metrics per image."""
    sphere_segmentation_kwargs = sphere_segmentation_kwargs or {}

    rows = []
    for tif_path in sorted(Path(validation_dir).glob("*.tif")):
        item = load_validation_item(tif_path, crop_banner=crop_banner)
        if item is None:
            continue
        image, ground_truth = item

        _, _, filtered_df = sphere_segmentation(
            image, mask_generator, nanometer_per_pixel=nanometer_per_pixel,
            **sphere_segmentation_kwargs,
        )

        metrics = evaluate_image(
            filtered_df, ground_truth, image.shape,
            nanometer_per_pixel=nanometer_per_pixel, filter_border=filter_border,
        )
        metrics["img_name"] = tif_path.name
        rows.append(metrics)

    return pd.DataFrame(rows)


def compare_models_wilcoxon(base_df, finetuned_df, metric="mean_abs_radius_error"):
    """Paired Wilcoxon signed-rank test between two per-image metric tables,
    matched by img_name. Standalone scipy.stats.wilcoxon call - unrelated to
    particle_stats.py's Welch's t-test."""
    merged = base_df.merge(finetuned_df, on="img_name", suffixes=("_base", "_finetuned"))
    base_values = merged[f"{metric}_base"]
    finetuned_values = merged[f"{metric}_finetuned"]

    result = wilcoxon(finetuned_values, base_values)

    return {
        "metric": metric,
        "n_pairs": len(merged),
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "median_base": base_values.median(),
        "median_finetuned": finetuned_values.median(),
    }

"""
Unit tests for particle_stats.py, the module that turns particle-size
measurements into a difference-of-means statistic (mean radius, confidence
interval, significance) between a reference sample and a test sample.

difference_of_means_statistic() implements Welch's t-test by hand (unequal
variances, Welch-Satterthwaite degrees of freedom). Rather than hand-computing
expected numbers, these tests check it against scipy.stats.ttest_ind's own
Welch's t-test (equal_var=False) as the reference implementation.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from particle_stats import difference_of_means_statistic, remove_outliers_df


def make_samples(seed, sample_mean, sample_std, sample_n, ref_mean, ref_std, ref_n):
    rng = np.random.default_rng(seed)
    sample = pd.Series(rng.normal(loc=sample_mean, scale=sample_std, size=sample_n), name="sample")
    reference = pd.Series(rng.normal(loc=ref_mean, scale=ref_std, size=ref_n), name="reference")
    return sample, reference


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_difference_of_means_matches_scipy_welch_ttest(alpha):
    sample, reference = make_samples(42, 52, 5, 40, 48, 6, 35)

    result = difference_of_means_statistic(reference.copy(), sample.copy(), diameter=False, alpha=alpha)

    scipy_result = stats.ttest_ind(sample, reference, equal_var=False)
    scipy_ci = scipy_result.confidence_interval(confidence_level=1 - alpha)

    assert result["thickness"] == pytest.approx(sample.mean() - reference.mean())
    assert result["p_value"] == pytest.approx(scipy_result.pvalue)
    assert result["DOF"] == pytest.approx(scipy_result.df)
    assert result["CI_lower"] == pytest.approx(scipy_ci.low)
    assert result["CI_upper"] == pytest.approx(scipy_ci.high)


def test_difference_of_means_flags_significant_when_scipy_pvalue_is_below_alpha():
    # Well-separated distributions -> a real, detectable difference.
    sample, reference = make_samples(1, 80, 3, 30, 40, 3, 30)
    alpha = 0.05

    result = difference_of_means_statistic(reference.copy(), sample.copy(), diameter=False, alpha=alpha)
    scipy_result = stats.ttest_ind(sample, reference, equal_var=False)

    assert scipy_result.pvalue < alpha
    assert result["significance"] == "Significant"


def test_difference_of_means_flags_not_significant_when_scipy_pvalue_exceeds_alpha():
    # Same distribution for both groups -> no real difference to detect.
    sample, reference = make_samples(2, 50, 5, 30, 50, 5, 30)
    alpha = 0.05

    result = difference_of_means_statistic(reference.copy(), sample.copy(), diameter=False, alpha=alpha)
    scipy_result = stats.ttest_ind(sample, reference, equal_var=False)

    assert scipy_result.pvalue > alpha
    assert result["significance"] == "Not significant"


def test_difference_of_means_diameter_flag_matches_scipy_on_halved_values():
    # diameter=True should be equivalent to running the radius-based (diameter=False)
    # test on values that have already been halved.
    sample, reference = make_samples(3, 100, 8, 25, 90, 8, 25)
    alpha = 0.1

    result = difference_of_means_statistic(reference.copy(), sample.copy(), diameter=True, alpha=alpha)
    scipy_result = stats.ttest_ind(sample / 2, reference / 2, equal_var=False)
    scipy_ci = scipy_result.confidence_interval(confidence_level=1 - alpha)

    assert result["p_value"] == pytest.approx(scipy_result.pvalue)
    assert result["CI_lower"] == pytest.approx(scipy_ci.low)
    assert result["CI_upper"] == pytest.approx(scipy_ci.high)


def test_remove_outliers_df_nans_out_planted_extreme_value():
    rng = np.random.default_rng(7)
    normal_vals = rng.normal(loc=50, scale=2, size=20)
    col_with_outlier = np.append(normal_vals, [500.0])  # 21 values, 1 extreme outlier
    df = pd.DataFrame({"with_outlier": col_with_outlier})

    cleaned = remove_outliers_df(df)

    outlier_index = len(col_with_outlier) - 1
    assert pd.isna(cleaned.loc[outlier_index, "with_outlier"])
    # The bulk of the (non-outlier) values should survive untouched.
    assert cleaned["with_outlier"].notna().sum() >= len(normal_vals) - 2


def test_remove_outliers_df_leaves_small_samples_untouched():
    # Below min_sample (default 10) -> outlier removal must be skipped entirely,
    # even though 1000 is clearly an outlier relative to 1, 2, 3.
    df = pd.DataFrame({"small_sample": [1.0, 2.0, 3.0, 1000.0]})

    cleaned = remove_outliers_df(df)

    pd.testing.assert_series_equal(cleaned["small_sample"], df["small_sample"])


def test_remove_outliers_df_leaves_column_unchanged_when_no_outliers_detected():
    # A constant column has no value outside its own 5th/95th percentile
    # bounds (they equal the constant itself), so nothing gets flagged.
    df = pd.DataFrame({"constant": [15.0] * 30})

    cleaned = remove_outliers_df(df)

    pd.testing.assert_series_equal(cleaned["constant"], df["constant"])

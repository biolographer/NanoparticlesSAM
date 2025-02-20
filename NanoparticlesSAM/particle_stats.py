import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
import scipy.stats as stats
from itertools import combinations
import math


def remove_outliers_df(df, lower_quantile=0.05, upper_quantile=0.95, min_sample=10):
    """
    Removes outliers in each column of a DataFrame based on Q5-Q95 but retains small samples 
    and keeps original data if no outliers are detected.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical data.
        lower_quantile (float): Lower quantile threshold (default: 5th percentile).
        upper_quantile (float): Upper quantile threshold (default: 95th percentile).
        min_sample (int): Minimum sample size per column below which no data is removed.

    Returns:
        pd.DataFrame: Cleaned DataFrame with outliers removed where necessary.
    """
    cleaned_df = df.copy()  # Make a copy to avoid modifying the original DataFrame

    for col in df.columns:
        series = df[col].dropna()  # Work with non-null values only

        if len(series) <= min_sample:  
            continue  # Skip outlier removal if the sample size is too small

        lower_bound = series.quantile(lower_quantile)
        upper_bound = series.quantile(upper_quantile)

        is_outlier = (series < lower_bound) | (series > upper_bound)

        if is_outlier.any():  # Only remove outliers if any are detected
            cleaned_df[col] = df[col].where(~is_outlier, other=pd.NA)  # Keep structure

    return cleaned_df

def plot_quantiles(df):
    cols = df.columns
    rows = math.ceil(len(cols) / 2)  
    fig, axs = plt.subplots(rows, 2, figsize=(8 * 2, 8 * rows))  

    for col, ax in zip(cols, axs.flat):
        col_data = df[col].dropna()  # Drop NaN values
        unique_vals = col_data.nunique()  # Count unique values
        
        if unique_vals < 2:  # Skip empty or single-value columns
            ax.axis('off')
            continue
        
        standardized_data = stats.zscore(col_data)  # Standardize data
        
        if np.isnan(standardized_data).all():  # If all values are NaN after z-score
            ax.axis('off')
            continue

        stats.probplot(standardized_data, dist='norm', plot=ax)  
        ax.set_title(f'Q-Q plot for: {col}')

    # Hide unused subplots
    for i in range(len(cols), len(axs.flat)):
        axs.flat[i].axis('off')

    plt.tight_layout()
    plt.show()

# Calculate the standard error of the difference between means
# with a standard error propagation for subtraction SE=sqrt(SE1**2/n2 + SE2**2/n2)
def SE(x, y):
    n1 = len(x)
    n2 = len(y)
    std1 = x.std(ddof=1)
    std2 = y.std(ddof=1)
    return np.sqrt((std1**2 / n1) + (std2**2 / n2))

def SP(x, y):
    n1 = len(x)
    n2 = len(y)
    std1 = x.std()
    std2 = y.std()
    return np.sqrt( ((n1 - 1) * std1**2 + (n2 -1) * std2**2 )/ (n1 + n2 - 2) )

# Calculate degrees of freedom using Welch-Satterthwaite equation
def WS_dof(x, y):
    n1 = len(x)
    n2 = len(y)
    std1 = x.std(ddof=1)
    std2 = y.std(ddof=1)
    return (((std1**2 / n1) + (std2**2 / n2))**2 / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1)))

# Margin of error
def ME(t, SE):
    return (t * SE)

# Confidence interval for the difference between medians
def CE(c1, c2, me):
    mean1 = c1.mean()
    mean2 = c2.mean()
    difference = mean1 - mean2
    ci_lower = difference - me
    ci_upper = difference + me
    return (ci_lower, ci_upper)

def CE_median(c1, c2, me):
    median1 = c1.median()
    median2 = c2.median()
    difference = median1 - median2
    ci_lower = difference - me
    ci_upper = difference + me
    return (ci_lower, ci_upper)



def difference_of_means_statistic(reference_particles, sample, diameter=False, alpha= 0.1):
    """
    Performs a difference of means analysis to give a confidence interval between mean radii and whether,
    the samples are statistically significantly different.

    Args:
        reference_particles (pandas.series): Array with particle sizes of the reference particles.
        sample (pandas.series): The sample that's compared to the reference.
        diameter (bool): Set to true if the values in the arrays correspond to the diameter. Otherwise radius is assumed.
        alpha (float): significance level
    Returns:
        pandas dataframe: row for 
            - difference of means
            - Standard error of difference of means (SEM)
            - confidence interval upper
            - confidence intervall lower
            - degrees of freedom, calculated by Welch-Satterthwaite equation
            - p-value
            - Significance level
            - N_ref = # measurements of reference
            - N_sample = # measurements of sample
    """
    if diameter:
        # switch to radius
        reference_particles /= 2
        sample /= 2

    sample_name = sample.name
    reference_name = reference_particles.name

    # calculate difference of means
    diff_means = sample.mean() - reference_particles.mean()
    #standard error
    se = SE(sample, reference_particles)
    # degrees of freedom
    dof = WS_dof(sample, reference_particles)

    # critical t-value and margin of error at the significance level
    t_critical = t.ppf(1 - alpha/2, dof)
    significance_level = (1-alpha/2)*100
    me = t_critical * se

    #confidence intervall
    ce_lower, ce_upper = CE(sample, reference_particles, me)

    # p-value calculation at 
    z_stat = (sample.mean() - reference_particles.mean()) / se
    p_value = 2 * (1 - t.cdf(abs(z_stat), dof))

    if (ce_lower > 0 and ce_upper > 0) or (ce_lower < 0 and ce_upper < 0):
        significant = 'Significant'
    else:
        significant = 'Not significant'


    result_dict = {'thickness': diff_means, 
                   'CI_lower': ce_lower,
                   'CI_upper': ce_upper,
                   'p_value': p_value,
                   'significance': significant,
                   'significance_level': significance_level,
                   'SEM': se,
                   'DOF': dof,
                   'N_sample': len(sample),
                   'N_ref': len(reference_particles)
                   }
    
    return result_dict


def analyze_samples(df, reference_column, diameter=False, alpha=0.1):
    reference_particles = df[reference_column]
    results = {}

    for col in df.columns:

        # Skip reference column
        if col == reference_column:
            continue  
        
        sample = df[col]
        result_dict = difference_of_means_statistic(reference_particles.dropna(), sample.dropna(), diameter, alpha)
        results[col] = result_dict  # Store result dict under the column name

    # Convert results to DataFrame
    result_df = pd.DataFrame.from_dict(results, orient='index')
    return result_df.T
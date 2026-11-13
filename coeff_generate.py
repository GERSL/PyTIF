"""Process HLS observations using time-series-based image fusion (TIF).

The workflow matches Landsat (L30) and Sentinel-2 (S10) observations by date,
clusters the matched reflectance pairs, and fits robust per-band regression
models. It processes the selected tile in spatial chips and writes a CSV table
of model coefficients, cluster centroids, R-squared values, and quality
assurance flags for each chip. Pixel rows within each chip are processed
using the configured CPU worker count, which defaults to one.

The command-line interface is provided by Click. Use ``python tif.py --help``
to view the available options. The ``--debug`` option runs the built-in
single-pixel test-data workflow without searching for or reading HLS imagery.

Reference:
    Song, K., Zhu, Z., Qiu, S., Olofsson, P., Neigh, C. S., Ju, J., & Zhou, Q.
    (2025). TIF: A time-series-based image fusion algorithm. Remote Sensing of
    Environment, 331, 115035. https://doi.org/10.1016/j.rse.2025.115035

Contributors:
    Shi Qiu (University of Connecticut; shi.qiu@uconn.edu)
    Luofan Dong (University of Connecticut; luofan.dong@uconn.edu)
    Kexin Song (Yale University; MATLAB version; kexin.song@uconn.edu)

History:
    2026-11-15: Initial implementation. Shi Qiu

Last updated:
    2026-11-15

"""

# Create python environment using commands below:
# /home/shq19004/miniconda3/bin/conda create --name tifpy315 python=3.15 --dry-run --yes
# conda activate tifpy315
# conda install rasterio gdal -y
# conda install click -y
# conda install numpy -y
# conda install scikit-learn -y
# conda install pyarrow -y # for pandas to read parquet files, which is faster than csv
import os
import click
import random
import time
from functools import partial
import numpy as np
import pandas as pd
from hls import find_hls_images, read_hls_cube
from sklearn.cluster import KMeans
from utils import generate_tile_chips

def validate_chip_size(imagesize: int, chipsize: int) -> None:
    """
    Validates if the chip size is a multiple of 10, 20, and 30 meters (multiple of 60)
    and perfectly divides the total image size. If invalid, raises a ValueError
    with the closest valid suggestions.
    """
    # Check if chip size satisfies the 10, 20, 30 rules AND divides the image size perfectly
    if (chipsize % 10 != 0 or chipsize % 20 != 0 or chipsize % 30 != 0) or (imagesize % chipsize != 0):
        # Calculate all perfectly valid options up to the maximum image boundary
        all_valid = [x for x in range(60, imagesize + 1, 60) if imagesize % x == 0]
        
        # Fallback check if the image itself is completely incompatible with a 60m grid base
        if not all_valid:
            raise ValueError(
                f"Image size ({imagesize}) is invalid or too small. It cannot be divided "
                f"into any chip size that satisfies the 10, 20, and 30 meter constraints."
            )
        
        # Isolate the closest lower and higher valid constraints relative to the user's input
        under = [x for x in all_valid if x < chipsize]
        over = [x for x in all_valid if x > chipsize]
        
        closest_suggestions = []
        if under:
            closest_suggestions.append(f"{under[-1]} (smaller)")
        if over:
            closest_suggestions.append(f"{over[0]} (larger)")
            
        suggestions_str = " or ".join(closest_suggestions)
        
        raise ValueError(
            f"Invalid chip size ({chipsize}). It must be a multiple of 60 that divides the "
            f"image size ({imagesize}) perfectly. Try using a close valid size like: {suggestions_str}."
        )
    
def select_pairs(pixel_ts_l30, pixel_dates_l30, pixel_ts_s10, pixel_dates_s10, day_tolerance=16):
    """
    Find matched observations and return X, Y arrays for linear regression.
    
    Args:
        pixel_ts_l30: L30 surface reflectance time series (n_dates_l30, n_bands)
        pixel_dates_l30: L30 observation dates (n_dates_l30,)
        pixel_ts_s10: S10 surface reflectance time series (n_dates_s10, n_bands)
        pixel_dates_s10: S10 observation dates (n_dates_s10,)
        day_tolerance: Temporal threshold in days to match observations (default: 16)
    
    Returns:
        matched_data: Array of shape (n_matches, 3) where:
            - Column 0: L30 reflectance value (predictor)
            - Column 1: S10 reflectance value (response)
            - Column 2: Temporal distance between matched dates
    """
    pair_data = []
    pair_delta_t = []
    # Loop through each L30 observation and find the closest S10 observation within the day_tolerance
    for i, date_l30 in enumerate(pixel_dates_l30):
        # Calculate temporal differences between L30 observation and all S10 observations
        day_diffs = np.abs(pixel_dates_l30[i] - pixel_dates_s10)
        day_diffs = [td.days for td in day_diffs]  # Convert to days
        
        min_day_diff = np.min(day_diffs)
        # Find observations within the tolerance threshold
        if min_day_diff <= day_tolerance:
            close_id = np.where(day_diffs == min_day_diff)[0][0]
            # For each matched S10 observation, create a pair
            matched_l30 = pixel_ts_l30[i, :]  # Exclude fmask band
            matched_s10 = pixel_ts_s10[close_id, :]  # Exclude fmask band
            day_distance = day_diffs[close_id]
            pair_data.append([matched_l30, matched_s10])
            pair_delta_t.append(day_distance)
    
    # return paried data
    return np.array(pair_data), np.array(pair_delta_t)

def evaluate_gap_clusters(pair_data, maxK, n_reference=100, random_state=42):
    """
    Evaluate k-means clusters using the gap statistic.

    Parameters
    ----------
    pair_data : array-like, shape (n_pairs, 2)
        Array of matched observations. Each row represents a pair of L30 and S10 observations.
    maxK : int
        Maximum number of clusters to evaluate.
    n_reference : int, default=100
        Number of reference datasets. Follows MATLAB’s documented defaults: 100 reference datasets
    random_state : int, default=42
        Seed for reproducible results.

    Returns
    -------
        A dictionary containing:
        - optimal_k: The optimal number of clusters.
        - inspected_k: The number of clusters inspected.
        - gap_values: The gap values for each number of clusters.
        - standard_errors: The standard errors for each number of clusters.
    """
    _, _, nbands = pair_data.shape
    # merge the pair_data into a single array for clustering
    data = np.hstack((pair_data[:, 0], pair_data[:, 1]))  # shape (n_pairs, 2)

    rng = np.random.default_rng(random_state)

    # Rotate the centered data into principal-component coordinates.
    mean = data.mean(axis=0)
    _, _, components = np.linalg.svd(data - mean, full_matrices=False)
    rotated = (data - mean) @ components.T
    lower = rotated.min(axis=0)
    upper = rotated.max(axis=0)

    k_values = np.arange(1, maxK + 1)
    gaps = np.empty(maxK)
    standard_errors = np.empty(maxK)
    models = []
    for i, k in enumerate(k_values):
        # model = KMeans(n_clusters=k, n_init=10, random_state=random_state) # this produced same results as MATLAB's kmeans
        model = KMeans(n_clusters=k, n_init=1, init='k-means++', tol=1e-4, random_state=random_state)
        model.fit(data)
        models.append(model)

        if model.inertia_ <= 0:
            raise ValueError(f"Within-cluster dispersion is zero for k={k}.")

        log_w = np.log(model.inertia_)
        reference_log_w = np.empty(n_reference)

        for b in range(n_reference):
            # Sample within a box aligned with the principal components.
            reference_rotated = rng.uniform(lower, upper, size=rotated.shape)
            reference = reference_rotated @ components + mean

            # reference_model = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 2**31 - 1)),) # this produced same results as MATLAB's kmeans
            reference_model = KMeans(n_clusters=k, n_init=1, init='k-means++', tol=1e-4, random_state=int(rng.integers(0, 2**31 - 1)),)
            reference_model.fit(reference)

            if reference_model.inertia_ <= 0:
                raise ValueError("Reference data have zero within-cluster dispersion.")

            reference_log_w[b] = np.log(reference_model.inertia_)

        gaps[i] = reference_log_w.mean() - log_w
        standard_errors[i] = (
            reference_log_w.std(ddof=1) * np.sqrt(1 + 1 / n_reference)
        )

    # MATLAB's default "globalMaxSE" selection rule.
    max_gap_index = np.argmax(gaps)
    threshold = gaps[max_gap_index] - standard_errors[max_gap_index]
    optimal_k = int(k_values[np.flatnonzero(gaps >= threshold)[0]])
    optimal_cluster = models[optimal_k - 1].predict(data) # Get the cluster labels for the optimal k
    optimal_centroid = models[optimal_k - 1].cluster_centers_ # Get the cluster centroids for the optimal k
    # reorganize the optimal_centroid to match pair_data's shape (k, 2, n_bands)
    optimal_centroid = optimal_centroid.reshape(optimal_k, 2, nbands) # n_bands is the number of bands in the pair_data
    return optimal_k, optimal_cluster, optimal_centroid, k_values, standard_errors, gaps

def load_test_data(test_file):
    # Load test data from testdata, following the MATLAB structure of the test data.
    from scipy.io import loadmat
    mat_data = loadmat(test_file)
    date_list_l30 = mat_data["SAMPLES"]["sdate_L"][0][0].flatten()
    date_list_s10 = mat_data["SAMPLES"]["sdate_S"][0][0].flatten()
    pixel_ts_l30 = mat_data["SAMPLES"]["line_t_L"][0][0] 
    pixel_ts_s10 = mat_data["SAMPLES"]["line_t_S"][0][0]
    
    # convert to datetime objects, by days, to satify the requirement of finding pairs
    from datetime import datetime, timedelta
    date_list_l30 = np.array(date_list_l30, dtype=np.float64) # convert the date_list to numpy array with float type
    date_list_l30 = [datetime(1, 1, 1) + timedelta(days=d - 366) for d in date_list_l30]
    date_list_s10 = np.array(date_list_s10, dtype=np.float64) # convert the date_list to numpy array with float type
    date_list_s10 = [datetime(1, 1, 1) + timedelta(days=d - 366) for d in date_list_s10] # does not matter if the year is not correct, because we only need the day of year to find the pairs, and the year is not important
    # convert to array
    date_list_l30 = np.array(date_list_l30)
    date_list_s10 = np.array(date_list_s10)

    pixel_ts_l30 = np.array(pixel_ts_l30, dtype=np.float64) # convert the pixel_ts to numpy array
    pixel_ts_s10 = np.array(pixel_ts_s10, dtype=np.float64) # convert the pixel_ts to numpy array
    # Keep the first 6 bands (columns :6) and the last column (-1) as fmask
    pixel_ts_l30 = np.hstack((pixel_ts_l30[:, :6]/10000.0, pixel_ts_l30[:, [-1]]))
    pixel_ts_s10 = np.hstack((pixel_ts_s10[:, :6]/10000.0, pixel_ts_s10[:, [-1]]))
    return pixel_ts_l30, date_list_l30, pixel_ts_s10, date_list_s10

def robust_fit(x, y, weight, max_iter=50, tol=None):
    """
    Fit a weighted robust linear regression using Tukey's bisquare.

    This standalone Python implementation was developed with
    assistance from ChatGPT (OpenAI), November 2026.

    The implementation was developed by studying the following
    MATLAB functions and code:

    - ``fitlm``: linear model fitting with ``'RobustOpts', 'on'``
      and prior observation ``'Weights'``.
    - ``statrobustfit``: weighted robust fitting using iteratively
      reweighted least squares, leverage-adjusted residuals, and
      the bisquare weight function.
    - ``statrobustsigma``: estimation of the robust residual
      standard deviation using the bisquare function and its
      numerical derivative.
    - ``LinearModel.postFit``: calculation of robust SSE and SST
      used to obtain ordinary R².

    The implementation was compared with one MATLAB ``fitlm``
    example. Further comparisons are needed to establish numerical
    equivalence across datasets and edge cases.

    Parameters
    ----------
    x : array-like
        Predictor values.
    y : array-like
        Response values.
    weight : array-like
        Prior observation weights, such as temporal-distance weights.
    max_iter : int, default=50
        Maximum number of robust fitting iterations.
    tol : float or None, default=None
        Relative convergence tolerance. If None, use the MATLAB
        default sqrt(machine epsilon).

    Returns
    -------
    intercept : float
        Fitted intercept.
    slope : float
        Fitted slope.
    r_squared : float
        Estimated MATLAB-style ordinary R².
    robust_weights : ndarray
        Final bisquare weights for the valid observations.
    converged : bool
        Whether the coefficient iterations converged.

    Notes
    -----
    Invalid observations and observations with zero prior weight are
    removed. Rank-deficient fits raise an error.
    """

    # ------------------------------------------------------------
    # Prepare and validate input data.
    # ------------------------------------------------------------
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    weight = np.asarray(weight, dtype=float).reshape(-1)


    if tol is None:
        tol = np.sqrt(np.finfo(float).eps)

    n = len(y)
    p = 2  # Intercept and slope
    dfe = n - p

    if dfe <= 0:
        raise ValueError("At least three valid observations are required.")

    # Model: y = intercept + slope * x
    design = np.column_stack((np.ones(n), x))

    # ------------------------------------------------------------
    # Initial weighted least-squares fit.
    # ------------------------------------------------------------
    sqrt_weight = np.sqrt(weight)
    weighted_design = design * sqrt_weight[:, np.newaxis]
    weighted_y = y * sqrt_weight

    if np.linalg.matrix_rank(weighted_design) < p:
        raise ValueError("The weighted design matrix is rank deficient.")

    beta = np.linalg.lstsq(
        weighted_design,
        weighted_y,
        rcond=None,
    )[0]

    # Retain the initial fit for the final error-scale estimate.
    beta_initial = beta.copy()

    # Leverage from the prior-weighted design matrix.
    q, _ = np.linalg.qr(weighted_design, mode="reduced")
    leverage = np.minimum(np.sum(q**2, axis=1), 0.9999)
    leverage_adjustment = np.sqrt(1.0 - leverage)

    # MATLAB's lower bound for residual scale.
    tiny_s = 1e-6 * np.std(weighted_y, ddof=1)

    if tiny_s == 0:
        tiny_s = 1.0

    tuning_constant = 4.685
    robust_weights = np.ones(n)
    converged = False

    def bisquare(values):
        """Tukey bisquare weight function."""
        result = np.zeros_like(values)
        inside = np.abs(values) < 1.0
        result[inside] = (1.0 - values[inside] ** 2) ** 2
        return result

    # ------------------------------------------------------------
    # Iteratively reweighted least squares.
    # ------------------------------------------------------------
    for _ in range(max_iter):
        # Residuals on the original response scale.
        residuals = y - design @ beta

        # Adjust residuals using leverage.
        adjusted_residuals = residuals / leverage_adjustment

        # MATLAB madsigma excludes the smallest p - 1 absolute
        # adjusted residuals.
        absolute_residuals = np.sort(np.abs(adjusted_residuals))
        mad_s = np.median(absolute_residuals[p - 1:]) / 0.6745

        scale = max(mad_s, tiny_s)
        u = adjusted_residuals / (tuning_constant * scale)

        # Calculate the robust weights.
        robust_weights = bisquare(u)

        # Combine prior and robust weights.
        combined_weight = weight * robust_weights
        sqrt_combined = np.sqrt(combined_weight)

        robust_design = design * sqrt_combined[:, np.newaxis]
        robust_y = y * sqrt_combined

        if np.linalg.matrix_rank(robust_design) < p:
            raise ValueError(
                "Robust weighting produced a rank-deficient fit."
            )

        beta_new = np.linalg.lstsq(
            robust_design,
            robust_y,
            rcond=None,
        )[0]

        # MATLAB's relative coefficient convergence criterion.
        difference = np.abs(beta_new - beta)
        allowed_difference = tol * np.maximum(
            np.abs(beta_new),
            np.abs(beta),
        )

        beta = beta_new

        if np.all(difference <= allowed_difference):
            converged = True
            break

    # if not converged:
    #     warnings.warn(
    #         "Robust regression reached the iteration limit.",
    #         RuntimeWarning,
    #         stacklevel=2,
    #     )

    # ------------------------------------------------------------
    # Calculate robust error scale.
    # ------------------------------------------------------------
    residuals = y - design @ beta
    adjusted_residuals = residuals / leverage_adjustment

    absolute_residuals = np.sort(np.abs(adjusted_residuals))
    mad_s = np.median(absolute_residuals[p - 1:]) / 0.6745
    scale = max(mad_s, tiny_s)

    # Initial-fit error estimate on the original response scale.
    initial_residuals = y - design @ beta_initial
    ols_sigma = np.linalg.norm(initial_residuals) / np.sqrt(dfe)

    # MATLAB uses a special branch when all robust weights are
    # effectively zero or one.
    binary_tolerance = np.sqrt(np.finfo(float).eps)

    binary_weights = np.all(
        (robust_weights < binary_tolerance)
        | (robust_weights > 1.0 - binary_tolerance)
    )

    if binary_weights:
        included = robust_weights > 1.0 - binary_tolerance
        included_dfe = np.count_nonzero(included) - p

        if included_dfe <= 0:
            raise ValueError(
                "Insufficient observations to estimate robust error."
            )

        # This MATLAB branch uses residuals with prior weights.
        weighted_residuals = residuals * sqrt_weight

        robust_sigma = (
            np.linalg.norm(weighted_residuals[included])
            / np.sqrt(included_dfe)
        )

    else:
        # Translation of MATLAB's statrobustsigma.
        st = scale * tuning_constant
        u = adjusted_residuals / st

        phi = u * bisquare(u)

        # MATLAB estimates the derivative numerically.
        delta = 0.0001
        u_lower = u - delta
        u_upper = u + delta

        phi_lower = u_lower * bisquare(u_lower)
        phi_upper = u_upper * bisquare(u_upper)

        dphi = (phi_upper - phi_lower) / (2.0 * delta)

        m1 = np.mean(dphi)
        m2 = np.sum((1.0 - leverage) * phi**2) / dfe

        if np.isclose(m1, 0.0):
            raise ValueError(
                "Cannot estimate robust error: mean derivative is near zero."
            )

        correction = 1.0 + (p / n) * (1.0 - m1) / m1

        robust_sigma = correction * np.sqrt(m2) * st / m1

    # Shrink toward the initial-fit estimate when appropriate.
    sigma = max(
        robust_sigma,
        np.sqrt(
            (
                ols_sigma**2 * p**2
                + robust_sigma**2 * n
            )
            / (p**2 + n)
        ),
    )

    # ------------------------------------------------------------
    # Calculate MATLAB-style ordinary R².
    # ------------------------------------------------------------
    mse = sigma**2
    sse = dfe * mse

    y_pred = design @ beta
    y_mean = np.average(y, weights=weight)

    ssr = np.sum(weight * (y_pred - y_mean) ** 2)
    sst = ssr + sse

    r_squared = ssr / sst if sst > 0 else np.nan

    return (
        float(beta[0]),
        float(beta[1]),
        float(r_squared),
        robust_weights,
        converged,
    )

def fit_linear_model(pair_data, pair_delta_t, idx_cluster):
    nbands = pair_data.shape[2]  # number of bands
    coeffs = np.zeros([nbands, 2])  # to store the intercepts for each cluster and band
    r_squareds = np.zeros([nbands])  # to store the R² for each cluster and band
    for ib in range(nbands):
        # Extract the data for the current band
        X = pair_data[idx_cluster, 0, ib]  # L30 reflectance values
        Y = pair_data[idx_cluster, 1, ib]  # S10 reflectance values
        D = pair_delta_t[idx_cluster]  # Temporal distances of the paired observations
        # Fair temporal weights.
        W = 1.0 / (1.0 + np.abs(D))
        intercept, slope, r_squared, _, converged = (robust_fit(X, Y, W))
        # Store the fitted coefficients for the current band.
        coeffs[ib, 0] = intercept
        coeffs[ib, 1] = slope
        r_squareds[ib] = r_squared
    return coeffs, r_squareds, converged

def filter_invalid(pixel_ts, pixel_dates):
    """Filter out invalid observations based on fmask and reflectance range."""
    idx_valid = pixel_ts[:, -1] < 2  # fmask < 2, valid pixels
    pixel_ts = pixel_ts[:, :-1]  # remove the fmask band, which is not used anymore
    idx_valid &= (
        (pixel_ts.min(axis=1) > 0)
        & (pixel_ts.max(axis=1) < 1.0)
    )  # reflectance values are within the valid range
    return pixel_ts[idx_valid], pixel_dates[idx_valid]

def process_pixel(
    coordinates,
    cube_l30,
    cube_s10,
    date_list_l30,
    date_list_s10,
    bands,
    start_row,
    start_col,
):
    """Process one S10 pixel and return its coefficient records."""
    row, col = coordinates
    pixel_results = []
    pixel_ts_l30 = cube_l30[:, :, row // 3, col // 3]
    pixel_ts_s10 = cube_s10[:, :, row, col]

    # Filter out invalid pixels based on fmask and reflectance range
    pixel_ts_l30, date_list_l30 = filter_invalid(pixel_ts_l30, date_list_l30)
    pixel_ts_s10, date_list_s10 = filter_invalid(pixel_ts_s10, date_list_s10)

    pair_data, pair_delta_t = select_pairs(
        pixel_ts_l30,
        date_list_l30,
        pixel_ts_s10,
        date_list_s10,
    )
    if pair_data.shape[0] < 3:
        for band in bands:
            pixel_results.append({
                "row": start_row + row,
                "col": start_col + col,
                "count": pair_data.shape[0],
                "cluster": np.nan,
                "band": band.lower(),
                "intercept": np.nan,
                "slope": np.nan,
                "centroid_l30": np.nan,
                "centroid_s10": np.nan,
                "r_squared": np.nan,
                "QA": 0,
            })
        return row, col, pixel_results

    optimal_k, optimal_cluster, optimal_centroid, _, _, _ = (
        evaluate_gap_clusters(pair_data, maxK=3)
    )

    for ik in range(optimal_k):
        count_pairs = np.sum(optimal_cluster == ik)
        if count_pairs < 3:
            for band in bands:
                pixel_results.append({
                    "row": start_row + row,
                    "col": start_col + col,
                    "count": count_pairs,
                    "cluster": ik + 1,
                    "band": band.lower(),
                    "intercept": np.nan,
                    "slope": np.nan,
                    "centroid_l30": np.nan,
                    "centroid_s10": np.nan,
                    "r_squared": np.nan,
                    "QA": 0,
                })
            continue
        idx_cluster = optimal_cluster == ik
        coeffs, r_squareds, converged = fit_linear_model(
            pair_data,
            pair_delta_t,
            idx_cluster
        )
        
        qa_flag = 2 if not converged else 3

        for ib, band in enumerate(bands):
            pixel_results.append({
                "row": start_row + row,
                "col": start_col + col,
                "count": count_pairs,
                "cluster": ik + 1,
                "band": band.lower(),
                "intercept": coeffs[ib, 0],
                "slope": coeffs[ib, 1],
                "centroid_l30": optimal_centroid[ik, 0, ib],
                "centroid_s10": optimal_centroid[ik, 1, ib],
                "r_squared": r_squareds[ib],
                "QA": qa_flag,
            })
    return row, col, pixel_results

@click.command()
@click.option('--tile', default="T18TYM", help='HLS tile name')
@click.option('--bands', default="Blue,Green,Red,NIR,SWIR1,SWIR2", help='Comma-separated list of bands to process')
@click.option('--l30', default="/gpfs/sharedfs1/zhulab/Shi/ProjectHLS10/TIF/Data/L30/T18TYM", help='Directory containing L30 images')
@click.option('--s10', default="/gpfs/sharedfs1/zhulab/Shi/ProjectHLS10/TIF/Data/S10/T18TYM", help='Directory containing S10 images')
@click.option('--output', default="/gpfs/sharedfs1/zhulab/Shi/ProjectHLS10/TIF/Output/TIFCoeff/T18TYM", help='Output directory for fused images')
@click.option('--imagesize', default=10980, help='Size of the images to process, e.g., 10980 for 10980 by 10980 pixels in 10 meters')
@click.option('--chipsize', default=180, help='Chip size for processing the images per core, e.g., 180 for 180 by 180 pixels in 10 meters per core')
@click.option('--ji', default=1, help='Job ID to use for processing, e.g., 1, 2, 3, etc.')
@click.option('--jn', default=1, help='Total number of jobs to split the processing into, e.g., 12 for 12 jobs')
@click.option('--cp', default=1, help='Number of CPU cores per job to use for processing, e.g., 12 for 12 cores')
@click.option('--debug', default=False, help='Run the built-in single-pixel test-data workflow.')
def main(tile, bands, l30, s10, output, imagesize, chipsize, ji, jn, cp, debug):
    # Recording the start time of the processing
    start_time = time.time()
    # convert bands string to list
    bands = [band.strip() for band in bands.split(",")]

    print("> Basic information")
    print(f">> Tile: {tile}")
    print(f">> Job: {ji} of {jn} ({cp} CPU cores)")
    print(f">> Bands: {', '.join(bands)}")
    print(f">> L30 input directory: {l30}")
    print(f">> S10 input directory: {s10}")
    print(f">> Output directory: {output}")
    print(f">> Image size: {imagesize} pixels")
    print(f">> Chip size: {chipsize} pixels")
    print(f">> Mode: {'debug' if debug else 'standard'}")

    if debug:
        # Need to define the directory for the test data, which is used for debugging and testing the code.
        dir_data = "/gpfs/sharedfs1/zhulab/Shi/ProjectHLS10/TIF/CodeMatlab/TestData/13TCF/CalibrationSamplesTS"
        dir_out = dir_data + "TIFCoeffDebug"
        # Force the bands to the default set for debug mode, as the test data is based on these bands.
        bands = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
        
        # Search for the test data files with .mat
        import glob
        test_data_files = glob.glob(os.path.join(dir_data, "*30to10*.mat"))
        test_data_files.sort()  # Sort the files for consistent processing order
        # NEED to Sort the matlab code in Line 144, runTIF_par_clb_v4.m, to keep same order, using codes bellows:
        # [~, sortedIdx] = sort({sampleTS.name});
        # sampleTS = sampleTS(sortedIdx);

        for test_file in test_data_files:
            print(">> Loading the built-in test data for one pixel.")
            pixel_ts_l30, pixel_dates_l30, pixel_ts_s10, pixel_dates_s10 = (
                load_test_data(test_file)
            )
            _, _, debug_results = process_pixel(
                (0, 0),
                pixel_ts_l30[:, :, np.newaxis, np.newaxis],
                pixel_ts_s10[:, :, np.newaxis, np.newaxis],
                pixel_dates_l30,
                pixel_dates_s10,
                bands,
                0,
                0,
            )
            os.makedirs(dir_out, exist_ok=True)
            debug_output_path = os.path.join(dir_out, f"debug_{os.path.basename(test_file).replace('.mat', '.csv')}")
            pd.DataFrame(debug_results).to_csv(debug_output_path, index=False)
            print(f">> Saved debug results to {debug_output_path}")
            

    # Check if the selected size is divisible by the image resolutions, 10, 20, and 30 meters, 
    # and at the same time, check if image size is divisible by chip size, if not, raise an error
    validate_chip_size(imagesize, chipsize)

    # Find images for L30 and S10 sensors
    image_list_l30, date_list_l30 = find_hls_images(l30, "L30")
    image_list_s10, date_list_s10 = find_hls_images(s10, "S10")
    print(f">> # of L30 images found: {len(image_list_l30)}")
    print(f">> # of S10 images found: {len(image_list_s10)}")
    # shuffle the image lists to ensure that the processing is balanced across jobs
    random.seed(ji)  # Set a seed for reproducibility for image and date list shuffling
    idx_random_l30 = list(range(len(image_list_l30)))
    random.shuffle(idx_random_l30)
    image_list_l30 = [image_list_l30[i] for i in idx_random_l30]
    date_list_l30 = np.array([date_list_l30[i] for i in idx_random_l30])
    idx_random_s10 = list(range(len(image_list_s10)))
    random.shuffle(idx_random_s10)
    image_list_s10 = [image_list_s10[i] for i in idx_random_s10]
    date_list_s10 = np.array([date_list_s10[i] for i in idx_random_s10])

    # Split the extent of the image into parts with the defined chipsize
    chip_list = generate_tile_chips(imagesize, chipsize) # chip name, starting row, staring column, ending row, ending column
    random.seed(42)  # Set a seed for reproducibility
    random.shuffle(chip_list) # Shuffle the chip_list to ensure that the processing is balanced across jobs
    
    # Assign this job its share of chips.
    chip_list = chip_list[ji-1::jn]
    
    print(f"> Processing {len(chip_list)} chips")

    # processing chip
    for chip in chip_list:
        chip_name, start_row, start_col, end_row, end_col = chip

        tif_output_path = os.path.join(output, f"{tile}_{chip_name}.parquet")
        if os.path.exists(tif_output_path):
            print(
                f">> Output for chip {chip_name} already exists"
            )
            continue

        print(
            f">> Processing rows {start_row}-{end_row}, "
            f"columns {start_col}-{end_col}"
        )
        
        tif_coeffs = [] # final results for this chip, which will be saved to a parquet file
 
        start_time_chip = time.time()
        # Read the time series of chips for L30 and S10 images
        cube_l30, cube_s10 = read_hls_cube(
            bands, image_list_l30, image_list_s10,
            start_row, start_col, end_row, end_col, cp=cp,
        )

        # define a partial function to process each pixel with the fixed parameters
        processer = partial(
            process_pixel,
            cube_l30=cube_l30,
            cube_s10=cube_s10,
            date_list_l30=date_list_l30,
            date_list_s10=date_list_s10,
            bands=bands,
            start_row=start_row,
            start_col=start_col,
        )
        for row in range(cube_s10.shape[2]):
            for col in range(cube_s10.shape[3]):
                #start_time_pixel = time.time()
                _, _, pixel_results = processer((row, col))
                tif_coeffs.extend(pixel_results)
                #print(f">> The {row+1}th in row & {col+1} processed in {(time.time() - start_time_pixel):.2f} seconds")

        os.makedirs(output, exist_ok=True) # Create the output directory if it does not exist

        # saving the results to parquet format for better performance and compression for Python
        pd.DataFrame(tif_coeffs).to_parquet(tif_output_path, engine="pyarrow", compression="snappy")
        print(
            f">> Saved TIF coefficients to {tif_output_path}"
        )
        print(f">> Elapsed processing time in {(time.time() - start_time_chip) / 60:.2f} minutes")
    print(f"> Total elapsed processing time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
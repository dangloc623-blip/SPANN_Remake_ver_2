"""Dip test / bimodality helpers (copied from model_origin.py, null simulation made cheaper)."""
import numpy as np
import scipy.stats


def calculate_bimodality_coefficient(series=None):
    """
    calculate BC-coefficient, higher BC coef means more likely to be a bimodal distribution
    """
    # Calculate skewness.
    # Correct for statistical sample bias.
    skewness = scipy.stats.skew(
        series,
        axis=None,
        bias=False,
    )
    # Calculate excess kurtosis.
    # Correct for statistical sample bias.
    kurtosis = scipy.stats.kurtosis(
        series,
        axis=None,
        fisher=True,
        bias=False,
    )
    # Calculate count.
    count = len(series)
    # Calculate count factor.
    count_factor = ((count - 1) ** 2) / ((count - 2) * (count - 3))
    # Count bimodality coefficient.
    coefficient = ((skewness**2) + 1) / (kurtosis + (3 * count_factor))
    # Return value.
    return coefficient


def dip(samples, num_bins=100, p=0.99, table=True, n_null=300):
    """
    dip test for distribution, to test is the distribution id multi-modality
    """
    samples = samples / np.abs(samples).max()
    pdf, idxs = np.histogram(samples, bins=num_bins)
    idxs = idxs[:-1] + np.diff(idxs)
    pdf = pdf / pdf.sum()

    cdf = np.cumsum(pdf, dtype=float)
    assert np.abs(cdf[-1] - 1) < 5e-2

    D = 0
    ans = 0
    check = False
    while True:
        gcm_values, gcm_contact_points = gcm_cal(cdf, idxs)
        lcm_values, lcm_contact_points = lcm_cal(cdf, idxs)

        d_gcm, gcm_diff = sup_diff(gcm_values, lcm_values, gcm_contact_points)
        d_lcm, lcm_diff = sup_diff(gcm_values, lcm_values, lcm_contact_points)

        if d_gcm > d_lcm:
            xl = gcm_contact_points[d_gcm == gcm_diff][0]
            xr = lcm_contact_points[lcm_contact_points >= xl][0]
            d = d_gcm
        else:
            xr = lcm_contact_points[d_lcm == lcm_diff][-1]
            xl = gcm_contact_points[gcm_contact_points <= xr][-1]
            d = d_lcm

        gcm_diff_ranged = np.abs(gcm_values[: xl + 1] - cdf[: xl + 1]).max()
        lcm_diff_ranged = np.abs(lcm_values[xr:] - cdf[xr:]).max()

        if d <= D or xr == 0 or xl == cdf.size:
            ans = D
            break
        else:
            D = max(D, gcm_diff_ranged, lcm_diff_ranged)

        cdf = cdf[xl : xr + 1]
        idxs = idxs[xl : xr + 1]
        pdf = pdf[xl : xr + 1]

    if table:
        p_threshold, p_value = p_table(p, ans, samples.size, n_null)
        if ans < p_threshold:
            check = True
        return ans, p_threshold, check, p_value

    return ans


def gcm_cal(cdf, idxs):
    local_cdf = np.copy(cdf)
    local_idxs = np.copy(idxs)
    gcm = [local_cdf[0]]
    contact_points = [0]
    while local_cdf.size > 1:
        distances = local_idxs[1:] - local_idxs[0]
        slopes = (local_cdf[1:] - local_cdf[0]) / distances
        slope_min = slopes.min()
        slope_min_idx = np.where(slopes == slope_min)[0][0] + 1
        gcm.append(local_cdf[0] + distances[:slope_min_idx] * slope_min)
        contact_points.append(contact_points[-1] + slope_min_idx)
        local_cdf = local_cdf[slope_min_idx:]
        local_idxs = local_idxs[slope_min_idx:]
    return np.hstack(gcm), np.hstack(contact_points)


def lcm_cal(cdf, idxs):
    values, points = gcm_cal(1 - cdf[::-1], idxs.max() - idxs[::-1])
    return 1 - values[::-1], idxs.size - points[::-1] - 1


def sup_diff(alpha, beta, contact_points):
    diff = np.abs(alpha[contact_points] - beta[contact_points])
    return diff.max(), diff


def p_table(p, ans, sample_size, n_samples):
    # speed-up: the null distribution only needs a modest number of simulations
    # (original: 10000 x randn(n_cells), which took tens of seconds)
    rng = np.random.default_rng(0)
    dip_sample = [dip(rng.standard_normal(sample_size), table=False) for _ in range(n_samples)]
    dips = np.hstack(dip_sample)
    dip_sample.append(ans)
    index = np.argsort(dip_sample)
    p_value = 1 - (np.argsort(index)[-1] + 1) / len(index)
    return np.percentile(dips, p * 100), p_value



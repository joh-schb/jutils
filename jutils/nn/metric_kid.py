# ALL CODE ADAPTED FROM: https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_kid.py
import torch
import numpy as np
from tqdm import tqdm


__all__ = [
    "kid_features_to_metric",
    "KEY_METRIC_KID_MEAN",
    "KEY_METRIC_KID_STD",
    "DEFAULTS",
]
# ===================================================================================================


KEY_METRIC_KID_MEAN = "kernel_distance_mean"
KEY_METRIC_KID_STD = "kernel_distance_std"

DEFAULTS = {
    "kid_subsets": 100,
    "kid_subset_size": 1000,
    "kid_kernel": "poly",
    "kid_kernel_poly_degree": 3,
    "kid_kernel_poly_gamma": None,
    "kid_kernel_poly_coef0": 1,
    "kid_kernel_rbf_sigma": 10,
    "verbose": False,
    "rng_seed": 0,
}


def get_kwarg(name, kwargs):
    return kwargs.get(name, DEFAULTS[name])


def vassert(truecond, message):
    if not truecond:
        raise ValueError(message)


def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est="unbiased"):
    assert mmd_est in ("biased", "unbiased", "u-statistic"), "Invalid value of mmd_est"

    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == "biased":
        mmd2 = (Kt_XX_sum + sum_diag_X) / (m * m) \
             + (Kt_YY_sum + sum_diag_Y) / (m * m) \
             - 2 * K_XY_sum / (m * m)  # fmt: skip
    else:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == "unbiased":
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    return mmd2


def kernel_poly(X, Y, **kwargs):
    degree = get_kwarg("kid_kernel_poly_degree", kwargs)
    gamma = get_kwarg("kid_kernel_poly_gamma", kwargs)
    coef0 = get_kwarg("kid_kernel_poly_coef0", kwargs)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K


def kernel_rbf(X, Y, **kwargs):
    sigma = get_kwarg("kid_kernel_rbf_sigma", kwargs)
    vassert(sigma is not None and sigma > 0, "kid_kernel_rbf_sigma must be positive")
    XX = np.sum(X**2, axis=1)
    YY = np.sum(Y**2, axis=1)
    XY = np.dot(X, Y.T)
    K = np.exp((2 * XY - np.outer(XX, np.ones(YY.shape[0])) - np.outer(np.ones(XX.shape[0]), YY)) / (2 * sigma**2))
    return K


def kernel_mmd(features_1, features_2, **kwargs):
    kernel = get_kwarg("kid_kernel", kwargs)
    vassert(kernel in ("poly", "rbf"), "Invalid KID kernel")
    kernel = {
        "poly": kernel_poly,
        "rbf": kernel_rbf,
    }[kernel]
    k_11 = kernel(features_1, features_1, **kwargs)
    k_22 = kernel(features_2, features_2, **kwargs)
    k_12 = kernel(features_1, features_2, **kwargs)
    return mmd2(k_11, k_12, k_22)


def kid_features_to_metric(features_1, features_2, **kwargs):
    """Compute Kernel Feature Distance (e.g. KID or KDD) between two sets of features.

    Args:
        features_1: Torch tensor of shape (N, D) containing features.
        features_2: Torch tensor of shape (N, D) containing features.
        kid_subsets: Number of subsets to use for KID computation.
        kid_subset_size: Size of each subset to use for KID computation.
        kid_kernel: Kernel to use for KID computation. One of 'poly' or 'rbf'.
        kid_kernel_poly_degree: Degree of the polynomial kernel (if kid_kernel is 'poly').
        kid_kernel_poly_gamma: Gamma parameter of the polynomial kernel (if kid_kernel is 'poly').
        kid_kernel_poly_coef0: Coefficient parameter of the polynomial kernel (if kid_kernel is 'poly').
        kid_kernel_rbf_sigma: Sigma parameter of the RBF kernel (if kid_kernel is 'rbf').
        verbose: Whether to print progress and results.
        rng_seed: Random seed for reproducibility.
    """
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]

    kid_subsets = get_kwarg("kid_subsets", kwargs)
    kid_subset_size = get_kwarg("kid_subset_size", kwargs)
    verbose = get_kwarg("verbose", kwargs)

    n_samples_1, n_samples_2 = len(features_1), len(features_2)
    vassert(
        n_samples_1 >= kid_subset_size and n_samples_2 >= kid_subset_size,
        f"KID subset size {kid_subset_size} cannot be smaller than the number of samples (input_1: {n_samples_1}, "
        f'input_2: {n_samples_2}). Consider using "kid_subset_size" kwarg or "--kid-subset-size" command line key to '
        f"proceed.",
    )

    features_1 = features_1.cpu().numpy()
    features_2 = features_2.cpu().numpy()

    mmds = np.zeros(kid_subsets)
    rng = np.random.RandomState(get_kwarg("rng_seed", kwargs))

    for i in tqdm(
        range(kid_subsets), disable=not verbose, leave=False, unit="subsets", desc="Kernel Inception Distance"
    ):
        f1 = features_1[rng.choice(n_samples_1, kid_subset_size, replace=False)]
        f2 = features_2[rng.choice(n_samples_2, kid_subset_size, replace=False)]
        o = kernel_mmd(f1, f2, **kwargs)
        mmds[i] = o

    out = {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }

    if verbose:
        print(f"Kernel Inception Distance: {out[KEY_METRIC_KID_MEAN]:.7g} ± {out[KEY_METRIC_KID_STD]:.7g}")

    return out

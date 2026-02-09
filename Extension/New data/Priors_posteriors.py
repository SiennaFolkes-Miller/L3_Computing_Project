import numpy as np
import astropy.units as u
from models import distance_modulus
from scipy.stats import norm
from scipy.stats import multivariate_normal
from cmb_directional import cmb_directional_logprior


def log_prior(theta, prior_type="flat_wide", flat_universe=False):

    Omega_L, Omega_k, H0 = theta
    Omega_m = 1.0 - Omega_L - Omega_k

    # Always enforce physicality
    if Omega_m < 0:
        return -np.inf

    if flat_universe and not np.isclose(Omega_k, 0.0):
        return -np.inf

    if prior_type == "flat_wide":
        if not (0.0 < Omega_L < 1.5):
            return -np.inf
        if not (-1.0 < Omega_k < 1.0):
            return -np.inf
        if not (40.0 < H0 < 100.0):
            return -np.inf
        return 0.0

    if prior_type == "flat_narrow":
        if not (0.6 < Omega_L < 0.8):
            return -np.inf
        if not (-0.2 < Omega_k < 0.2):
            return -np.inf
        if not (60.0 < H0 < 75.0):
            return -np.inf
        return 0.0

    if prior_type == "cmb_gaussian":
        logp = 0.0
        logp += norm.logpdf(Omega_L, 0.6847, 0.0073*10)
        logp += norm.logpdf(H0, 67.36, 0.54*10)
        logp += norm.logpdf(Omega_k, 0.0007, 0.0019*10)
        return logp

    if prior_type == "cmb_directional":
        # Standard deviations and correlation matrix
        sigma = np.array([0.01, 5.0, 0.01])
        rho = np.array([
            [1.0, 0.5, -0.7],
            [0.5, 1.0, -0.6],
            [-0.7, -0.6, 1.0]
        ])
        cov_cmb = np.outer(sigma, sigma) * rho
        icov_cmb = np.linalg.inv(cov_cmb)

        # Parameter vector
        p = np.array([Omega_L, H0, Omega_k])
        diff = p - np.array([0.6847, 67.36, 0.0007])
        logp = -0.5 * diff @ icov_cmb @ diff
        return logp
        #return cmb_directional_logprior(theta)

    raise ValueError(f"Unknown prior_type: {prior_type}")


def log_likelihood(theta, z, mu_obs, mu_err):
    """
    Gaussian SN Ia likelihood.
    theta = (Omega_L, Omega_k, H0)
    """
    Omega_L, Omega_k, H0_val = theta

    mu_model = distance_modulus(z, Omega_L, Omega_k, H0_val)

    return -0.5 * np.sum(((mu_obs - mu_model) / mu_err) ** 2)


def log_posterior(
    theta,
    z,
    mu_obs,
    mu_err,
    prior_type="flat_wide",
    flat_universe=False
):
    lp = log_prior(theta, prior_type=prior_type, flat_universe=flat_universe)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, z, mu_obs, mu_err)
import numpy as np
import astropy.units as u
from models import distance_modulus
from scipy.stats import norm
from scipy.stats import multivariate_normal
from CMB_directional_data import cmb_center, cmb_icov



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
        if not flat_universe:
            return -np.inf
        mu_H0 = 67.36
        mu_OmegaL = 1.0 - 0.3153 
        var_H0 = (0.54*5)**2
        var_OmegaL = (0.0073*5)**2
        logp  = norm.logpdf(H0, mu_H0, np.sqrt(var_H0))
        logp += norm.logpdf(Omega_L, mu_OmegaL, np.sqrt(var_OmegaL))
        return logp

    if prior_type == "cmb_directional":
    # Ensure flat universe
        if not flat_universe:
            return -np.inf

    # -----------------------
    # Clip parameters to avoid unphysical values
    # Adjust these bounds if needed
        if H0 <= 0 or H0 < 50 or H0 > 100:
            return -np.inf
        if Omega_L < 0 or Omega_L > 2:
            return -np.inf

    # Compute the prior (Gaussian with cmb_center and covariance)
        diff = np.array([H0, Omega_L]) - cmb_center

    # Check for NaN in diff or icov
        if not np.all(np.isfinite(diff)) or not np.all(np.isfinite(cmb_icov)):
            return -np.inf

        lp = -0.5 * diff @ cmb_icov @ diff

    # Ensure the prior value is finite
        if not np.isfinite(lp):
            return -np.inf

        return lp

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
import numpy as np
import astropy.units as u
from models import distance_modulus
from scipy.stats import norm


def log_prior(theta, prior_type="flat", flat_universe=False):

    Omega_L, Omega_k, H0 = theta

    # Physical boundaries (always enforced)
    if not (0.0 < Omega_L < 1.5):
        return -np.inf
    if not (-1.0 < Omega_k < 1.0):
        return -np.inf
    if not (40.0 < H0 < 100.0):
        return -np.inf
    Omega_m = 1.0 - Omega_L - Omega_k
    if Omega_m < 0.0:
        return -np.inf

    # Enforce flat universe if requested
    if flat_universe and not np.isclose(Omega_k, 0.0):
        return -np.inf

    # Flat prior = uniform in allowed range
    if prior_type == "flat":
        return 0.0

    # Gaussian prior (rough CMB-inspired)
    elif prior_type == "gaussian":
        # Gaussian prior on Omega_L
        mu_OL = 0.6847
        sigma_OL = 0.0073*100
        logp_OL = norm.logpdf(Omega_L, loc=mu_OL, scale=sigma_OL)

        # Gaussian prior on H0
        mu_H0 = 67.36
        sigma_H0 = 0.54*100
        logp_H0 = norm.logpdf(H0, loc=mu_H0, scale=sigma_H0)

        # Optional Gaussian prior on Omega_k
        mu_Ok = 0.0
        sigma_Ok = 0.05*100
        logp_Ok = norm.logpdf(Omega_k, loc=mu_Ok, scale=sigma_Ok)

        return logp_OL + logp_H0 + logp_Ok

    else:
        raise ValueError("prior_type must be 'flat' or 'gaussian'")


def log_likelihood(theta, z, mu_obs, mu_err, flat=False):
    if flat:
        Omega_L, H0_val = theta
        Omega_k = 0.0
    else:
        Omega_L, Omega_k, H0_val = theta

    # distance_modulus handles H0 units internally
    mu_model = distance_modulus(z, Omega_L, Omega_k, H0_val)
    
    return -0.5 * np.sum(((mu_obs - mu_model)/mu_err)**2)


def log_posterior(theta, z, mu_obs, mu_err, prior_type="flat", flat_universe=False):
    lp = log_prior(theta, prior_type=prior_type, flat_universe=flat_universe)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu_obs, mu_err)
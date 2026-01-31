import numpy as np
import astropy.units as u
from models import distance_modulus



def log_prior(theta, flat=False):
    """
    Priors on cosmological parameters.

    theta = [Omega_L, Omega_k, H0]
    """
    Omega_L, Omega_k, H0 = theta

    # Flat-universe run: force Omega_k = 0
    if flat:
        if not np.isclose(Omega_k, 0.0):
            return -np.inf

    # Physical priors
    if not (0.0 < Omega_L < 1.5):
        return -np.inf

    if not (-1.0 < Omega_k < 1.0):
        return -np.inf

    if not (40.0 < H0 < 100.0):
        return -np.inf

    # Derived matter density must be physical
    Omega_m = 1.0 - Omega_L - Omega_k
    if Omega_m < 0.0:
        return -np.inf

    return 0.0


def log_likelihood(theta, z, mu_obs, mu_err, flat=False):
    if flat:
        Omega_L, H0_val = theta
        Omega_k = 0.0
    else:
        Omega_L, Omega_k, H0_val = theta

    # distance_modulus handles H0 units internally
    mu_model = distance_modulus(z, Omega_L, Omega_k, H0_val)
    
    return -0.5 * np.sum(((mu_obs - mu_model)/mu_err)**2)


def log_posterior(theta, z, mu_obs, mu_err, flat=False):
    lp = log_prior(theta, flat=flat)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, z, mu_obs, mu_err)
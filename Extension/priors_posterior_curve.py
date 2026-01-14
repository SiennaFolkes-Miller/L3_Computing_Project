import numpy as np
from model_curve import m_model_curved  # your new curved m_model

# -------------------------------
# 1️⃣ Log-likelihood
# -------------------------------
def log_likelihood_m(theta, z, m_obs, sigma_m):
    """
    Log-likelihood for supernova magnitudes in curved ΛCDM.

    Parameters
    ----------
    theta : list or array
        [Omega_L, Omega_k, L]
    z : array
        Redshifts
    m_obs : array
        Observed magnitudes
    sigma_m : array
        Magnitude uncertainties
    """
    Omega_L, Omega_k, L = theta

    # Predict magnitudes
    m_pred = m_model_curved(z, Omega_L, Omega_k, L)

    # Gaussian log-likelihood
    chi2 = np.sum(((m_obs - m_pred) / sigma_m)**2)
    return -0.5 * chi2

# -------------------------------
# 2️⃣ Log-prior
# -------------------------------
def log_prior_m(theta):
    Omega_L, Omega_k, L = theta

    # Positive L
    if L <= 0:
        return -np.inf

    # Physical densities: 0 <= Omega_m = 1 - Omega_L - Omega_k <= 1
    Omega_m = 1.0 - Omega_L - Omega_k
    if Omega_m < 0 or Omega_m > 1:
        return -np.inf

    # Gaussian prior on Omega_L (from CMB)
    mu_OL = 0.6847
    sigma_OL = 0.0073
    lnP_OmegaL = -0.5 * ((Omega_L - mu_OL)/sigma_OL)**2

    return lnP_OmegaL

# -------------------------------
# 3️⃣ Log-posterior
# -------------------------------
def log_posterior_m(theta, z, m_obs, sigma_m):
    lp = log_prior_m(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_m(theta, z, m_obs, sigma_m)

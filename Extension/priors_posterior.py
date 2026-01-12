import numpy as np
from model_flat import m_model

# Assuming m_model is defined as you already have:
# def m_model(z, Omega_L, L):

def log_likelihood(theta, z, m_obs, sigma_m):
    """
    Log-likelihood for the supernova data.
    
    theta: [Omega_L, L]
    z: array of redshifts
    m_obs: observed magnitudes
    sigma_m: magnitude uncertainties
    """
    Omega_L, L = theta

    # Predict
    m_pred = m_model(z, Omega_L, L)

    # Gaussian log-likelihood
    chi2 = np.sum(((m_obs - m_pred) / sigma_m)**2)
    return -0.5 * chi2


def log_prior_flat(theta):
    Omega_L, L = theta
    
    # Flat prior on Omega_L
    if 0.0 <= Omega_L <= 1.5 and L > 0:
        return 0.0  # log(1)
    return -np.inf  # impossible

def log_prior_gauss(theta):
    Omega_L, L = theta
    
    # Gaussian prior on Omega_L
    mu = 0.7
    sigma = 0.1
    
    if L <= 0:
        return -np.inf
    
    lnP = -0.5 * ((Omega_L - mu) / sigma)**2
    return lnP


def log_posterior(theta, z, m_obs, sigma_m, prior="flat"):
    if prior == "flat":
        lp = log_prior_flat(theta)
    elif prior == "gauss":
        lp = log_prior_gauss(theta)
    else:
        raise ValueError("Unknown prior type")

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, z, m_obs, sigma_m)


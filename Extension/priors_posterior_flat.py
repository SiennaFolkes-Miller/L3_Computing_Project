import numpy as np
from model_flat import m_model
from model_flat import mu_model


# Assuming m_model is defined as you already have:
# def m_model(z, Omega_L, L):

def log_likelihood_m(theta, z, m_obs, sigma_m):
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


def log_prior_m(theta):
    Omega_L, L = theta
    
    # Flat prior on Omega_L
    if 0.0 <= Omega_L <= 1.5 and L > 0:
        return 0.0  # log(1)
    return -np.inf  # impossible

def log_prior_m(theta):
    """
    Prior: Gaussian on Omega_Lambda from CMB + flat on L.
    """
    Omega_L, L = theta
    
    # Gaussian prior on Omega_L from Planck-like constraint
    mu_OL = 0.6847    # mean
    sigma_OL = 0.0073     # 1σ width
    
    if L <= 0:
        return -np.inf  # L must be positive

    # Gaussian log-prior for Omega_L
    lnP_OmegaL = -0.5 * ((Omega_L - mu_OL)/sigma_OL)**2

    return lnP_OmegaL


def log_posterior_m(theta, z, m_obs, sigma_m):
    lp = log_prior_m(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood_m(theta, z, m_obs, sigma_m)



#new data

def log_prior_mu(theta):
    Omega_L = theta[0]
    if 0.0 <= Omega_L <= 1.5:
        return 0.0
    return -np.inf

def log_likelihood_mu(theta, z, mu_obs, sigma_mu):

    Omega_L = theta[0]  # only one parameter

    # Predict distance modulus
    mu_pred = mu_model(z, Omega_L)

    # Gaussian log-likelihood
    chi2 = np.sum(((mu_obs - mu_pred) / sigma_mu)**2)
    return -0.5 * chi2

def log_posterior_mu(theta, z, mu_obs, sigma_mu):
    lp = log_prior_mu(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood_mu(theta, z, mu_obs, sigma_mu)
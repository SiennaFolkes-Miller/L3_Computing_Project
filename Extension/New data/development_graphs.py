from new_data_splitting import z, mu, sigma_mu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from models import distance_modulus  # your model function

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 14

def plot_distance_modulus_with_residuals_fixed(z, mu_obs, mu_err, best_fit_theta, flat_universe=False):
    """
    Plots distance modulus vs redshift with best-fit model and normalized residuals.
    
    Parameters
    ----------
    z : array
        Redshift values
    mu_obs : array
        Observed distance modulus
    mu_err : array
        Observational errors
    best_fit_theta : array-like
        Best-fit parameters. If flat_universe: [Omega_L, H0], else [Omega_L, Omega_k, H0]
    flat_universe : bool
        Whether Omega_k = 0
    """
    if flat_universe:
        Omega_L, H0 = best_fit_theta
        Omega_k = 0.0
    else:
        Omega_L, Omega_k, H0 = best_fit_theta

    z_smooth = np.linspace(min(z), max(z), 500)
    mu_model_smooth = distance_modulus(z_smooth, Omega_L, Omega_k, H0)
    mu_model_obs = distance_modulus(z, Omega_L, Omega_k, H0)
    
    norm_residuals = (mu_obs - mu_model_obs) / mu_err

    # Plot
    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    # Main panel
    ax_main.errorbar(z, mu_obs, yerr=mu_err, fmt='o', ms=4, alpha = 0.5, color='black', label='Observed')
    ax_main.plot(z_smooth, mu_model_smooth, color='blue', lw=4, label='Best-fit')
    ax_main.legend(fontsize=LEGEND_SIZE)
    ax_main.set_ylabel(r"$\mu$", fontsize=AXIS_LABEL_SIZE)
    ax_main.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax_main.grid(False)

    # Residuals panel
    ax_resid.scatter(z, norm_residuals, s=15, alpha=0.5, color='blue')
    ax_resid.axhline(np.mean(norm_residuals), color='black', lw=3)
    ax_resid.axhline(np.mean(norm_residuals) + np.std(norm_residuals), color='grey', linestyle='--', lw=2)
    ax_resid.axhline(np.mean(norm_residuals) - np.std(norm_residuals), color='grey', linestyle='--', lw=2)
    ax_resid.set_ylabel("Normalised \n residuals", fontsize=AXIS_LABEL_SIZE)
    ax_resid.set_xlabel("Redshift $z$", fontsize=AXIS_LABEL_SIZE)
    ax_resid.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax_resid.grid(False)

    plt.tight_layout()
    plt.show()



def plot_residual_histograms_simple(z, mu_obs, mu_err, best_fit_flat, best_fit_curved):
    """
    Plot two histograms of normalized residuals: flat vs curved best-fit models.
    
    Parameters
    ----------
    z : array
        Redshift values
    mu_obs : array
        Observed distance modulus
    mu_err : array
        Observational errors
    best_fit_flat : [Omega_L, H0]
        Best-fit parameters for flat universe
    best_fit_curved : [Omega_L, Omega_k, H0]
        Best-fit parameters for curved universe
    """
    fig, axes = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    bins = np.linspace(-5, 5, 30)

    # Flat universe residuals
    Omega_L, H0 = best_fit_flat
    mu_flat = distance_modulus(z, Omega_L, 0.0, H0)
    res_flat = (mu_obs - mu_flat) / mu_err
    sns.histplot(res_flat, bins=bins, stat='density', color='green', ax=axes[0], label='Flat')
    mu_r, sigma_r = norm.fit(res_flat)
    x = np.linspace(-5, 5, 500)
    axes[0].plot(x, norm.pdf(x, mu_r, sigma_r), 'k--', lw=2,
                 label=f'Fit μ={mu_r:.3f}, \n σ={sigma_r:.3f}')
    print(sigma_r)
    axes[0].legend(fontsize=12)
    #axes[0].set_title("Flat Universe Residuals", fontsize=AXIS_LABEL_SIZE)
    axes[0].set_xlabel("Normalised residuals", fontsize=AXIS_LABEL_SIZE)
    axes[0].set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    axes[0].grid(False)

    # Curved universe residuals
    Omega_L, Omega_k, H0 = best_fit_curved
    mu_curved = distance_modulus(z, Omega_L, Omega_k, H0)
    res_curved = (mu_obs - mu_curved) / mu_err
    sns.histplot(res_curved, bins=bins, stat='density', color='blue', ax=axes[1], label='Curved')
    mu_r, sigma_r = norm.fit(res_curved)
    axes[1].plot(x, norm.pdf(x, mu_r, sigma_r), 'k--', lw=2,
                 label=f'Fit μ={mu_r:.3f}, \n σ={sigma_r:.3f}')
    print(sigma_r)
    axes[1].legend(fontsize=12)
    #axes[1].set_title("Curved Universe Residuals", fontsize=AXIS_LABEL_SIZE)
    axes[1].set_xlabel("Normalised residuals", fontsize=AXIS_LABEL_SIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    axes[1].set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)
    axes[1].grid(False)

    plt.tight_layout()
    plt.show()

best_fit_flat = [0.7076, 69.53]          # Omega_L, H0
best_fit_curved = [0.6992, 0.01317, 69.51]   # Omega_L, Omega_k, H0


# Distance modulus plot
plot_distance_modulus_with_residuals_fixed(z, mu, sigma_mu, best_fit_curved, flat_universe=False)

#residuals plot
plot_residual_histograms_simple(z, mu, sigma_mu, best_fit_flat, best_fit_curved)

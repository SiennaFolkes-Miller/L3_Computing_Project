import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from models import distance_modulus  # Your model function
from MCMC import run_supernova_mcmc
from new_data_splitting import z, mu, sigma_mu

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 14

def plot_distance_modulus_with_residuals(z, mu_obs, mu_err, posterior_samples, flat_universe=False):
    """
    Plots distance modulus vs redshift with 68% confidence bands
    and normalized residuals underneath.
    """

    if flat_universe:
        Omega_L_idx, H0_idx = 0, 1
        Omega_k_val = 0.0
    else:
        Omega_L_idx, Omega_k_idx, H0_idx = 0, 1, 2

    z_smooth = np.linspace(min(z), max(z), 500)

    # Compute model predictions
    mu_models = []
    for theta in posterior_samples:
        if flat_universe:
            mu_models.append(distance_modulus(z_smooth, theta[Omega_L_idx], 0.0, theta[H0_idx]))
        else:
            mu_models.append(distance_modulus(z_smooth, theta[Omega_L_idx], theta[Omega_k_idx], theta[H0_idx]))
    mu_models = np.array(mu_models)

    mu_best = np.median(mu_models, axis=0)
    mu_lo, mu_hi = np.percentile(mu_models, [16, 84], axis=0)

    # Residuals (using representative posterior draw)
    if flat_universe:
        mu_model_obs = distance_modulus(z, posterior_samples[0][Omega_L_idx], 0.0, posterior_samples[0][H0_idx])
    else:
        mu_model_obs = distance_modulus(
            z,
            posterior_samples[0][Omega_L_idx],
            posterior_samples[0][Omega_k_idx],
            posterior_samples[0][H0_idx]
        )

    norm_residuals = (mu_obs - mu_model_obs) / mu_err

    # Plot
    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    # Main panel
    ax_main.plot(z_smooth, mu_best, color='red', lw=4, label='Best-fit')
    ax_main.errorbar(
        z, mu_obs, yerr=mu_err,
        fmt='o', ms=4,
        color='blue', label='Observed'
    )

    ax_main.legend(fontsize=LEGEND_SIZE)
    ax_main.set_ylabel(r"$\mu$", fontsize=AXIS_LABEL_SIZE)
    ax_main.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax_main.grid(False)

    # Residuals panel
    ax_resid.scatter(z, norm_residuals, s=15, alpha=0.5, color='blue')
    ax_resid.axhline(np.mean(norm_residuals), color='black', lw=3)
    ax_resid.axhline(np.mean(norm_residuals) + np.std(norm_residuals),
                     color='grey', linestyle='--', lw=2)
    ax_resid.axhline(np.mean(norm_residuals) - np.std(norm_residuals),
                     color='grey', linestyle='--', lw=2)

    ax_resid.set_ylabel("Normalised \n residuals", fontsize=AXIS_LABEL_SIZE)
    ax_resid.set_xlabel("Redshift $z$", fontsize=AXIS_LABEL_SIZE)
    ax_resid.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax_resid.grid(False)

    plt.tight_layout()
    plt.show()




def compute_ppc_residuals(z, mu_obs, mu_err, posterior_samples, nsim=500, flat_universe=False):
    """
    Compute normalized residuals for posterior predictive checks.
    
    Parameters
    ----------
    z : array
        Redshift values
    mu_obs : array
        Observed distance modulus
    mu_err : array
        Observational errors
    posterior_samples : array
        Posterior samples from MCMC
    nsim : int
        Number of posterior samples to draw for PPC
    flat_universe : bool
        Whether to enforce Omega_k=0
    """
    # Compute median model from posterior
    if flat_universe:
        mu_med = np.median([distance_modulus(z, s[0], 0.0, s[1]) for s in posterior_samples], axis=0)
    else:
        mu_med = np.median([distance_modulus(z, s[0], s[1], s[2]) for s in posterior_samples], axis=0)

    # Collect residuals
    res_ppc = []
    idx = np.random.choice(len(posterior_samples), nsim, replace=False)
    for i in idx:
        theta = posterior_samples[i]
        if flat_universe:
            mu_model = distance_modulus(z, theta[0], 0.0, theta[1])
        else:
            mu_model = distance_modulus(z, theta[0], theta[1], theta[2])
        
        # Add observational noise
        mu_sim = np.random.normal(mu_model, mu_err)
        
        # Normalized residual
        res_ppc.append((mu_sim - mu_med) / mu_err)

    return np.array(res_ppc).flatten()

# ------------------------------------------------------
# Function 2: 2x2 grid of normalized residuals with PPC
# ------------------------------------------------------
def plot_residual_histograms(z, mu_obs, mu_err, samples_flat, samples_curved, nsim=200):
    """
    2x2 grid of normalized residual histograms with PPC.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)

    bins = np.linspace(-5, 5, 30)

    def median_model(z, samples, flat_universe=False, nsamples=500):
        idx = np.random.choice(len(samples), nsamples, replace=False)
        mu_models = []
        for i in idx:
            theta = samples[i]
            j = 0
            Omega_L = theta[j]; j += 1
            Omega_k = 0.0 if flat_universe else theta[j]
            if not flat_universe: j += 1
            H0 = theta[j]
            mu_models.append(distance_modulus(z, Omega_L, Omega_k, H0))
        return np.percentile(np.array(mu_models), 50, axis=0)

    # Curved real residuals
    mu_med_curved = median_model(z, samples_curved, flat_universe=False)
    norm_res_curved = (mu_obs - mu_med_curved) / mu_err
    sns.histplot(norm_res_curved, bins=bins, stat='density',
                 color='blue', ax=axes[0, 0], label='Observed (curved)')
    mu_r, sigma_r = norm.fit(norm_res_curved)
    x = np.linspace(-5, 5, 500)
    axes[0, 0].plot(x, norm.pdf(x, mu_r, sigma_r), 'k--', lw=2,
                    label=f'Fit μ={mu_r:.2f}, σ={sigma_r:.2f}')
    axes[0, 0].legend(fontsize=LEGEND_SIZE)
    axes[0, 0].set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)

    # Curved PPC
    res_ppc_curved = compute_ppc_residuals(z, mu_obs, mu_err,
                                           samples_curved, nsim=500, flat_universe=False)
    sns.histplot(res_ppc_curved, bins=bins, stat='density',
                 color='lightblue', ax=axes[0, 1], label='PPC (curved)')
    mu_r, sigma_r = norm.fit(res_ppc_curved)
    axes[0, 1].plot(x, norm.pdf(x, mu_r, sigma_r), 'k--', lw=2,
                    label=f'Fit μ={mu_r:.2f}, σ={sigma_r:.2f}')
    axes[0, 1].legend(fontsize=LEGEND_SIZE)
    axes[0, 1].set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)

    # Flat real residuals
    mu_med_flat = median_model(z, samples_flat, flat_universe=True)
    norm_res_flat = (mu_obs - mu_med_flat) / mu_err
    sns.histplot(norm_res_flat, bins=bins, stat='density',
                 color='green', ax=axes[1, 0], label='Observed (flat)')
    mu_r, sigma_r = norm.fit(norm_res_flat)
    axes[1, 0].plot(x, norm.pdf(x, mu_r, sigma_r), 'k--', lw=2,
                    label=f'Fit μ={mu_r:.2f}, σ={sigma_r:.2f}')
    axes[1, 0].legend(fontsize=LEGEND_SIZE)
    axes[1, 0].set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)

    # Flat PPC
    res_ppc_flat = compute_ppc_residuals(z, mu_obs, mu_err,
                                         samples_flat, nsim=500, flat_universe=True)
    sns.histplot(res_ppc_flat, bins=bins, stat='density',
                 color='lightgreen', ax=axes[1, 1], label='PPC (flat)')
    mu_r, sigma_r = norm.fit(res_ppc_flat)
    axes[1, 1].plot(x, norm.pdf(x, mu_r, sigma_r), 'k--', lw=2,
                    label=f'Fit μ={mu_r:.2f}, σ={sigma_r:.2f}')
    axes[1, 1].legend(fontsize=LEGEND_SIZE)
    axes[1, 1].set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)

    # Formatting
    for ax in axes.flatten():
        ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
        ax.grid(False)
        ax.set_title('')
    axes[1, 0].set_xlabel("Normalised residuals", fontsize=AXIS_LABEL_SIZE)
    axes[1, 1].set_xlabel("Normalised residuals", fontsize=AXIS_LABEL_SIZE)

    plt.tight_layout()
    plt.show()



# ------------------------------------------------------
# Example usage
# ------------------------------------------------------
samples_flat, _ = run_supernova_mcmc(
    z, mu, sigma_mu,
    prior_type="flat_wide",
    flat_universe=True,
    fix_H0=False,
    nwalkers=50,
    nsteps=1000,
    discard=200,
    output=False
)

samples_curved, _ = run_supernova_mcmc(
    z, mu, sigma_mu,
    prior_type="flat_wide",
    flat_universe=False,
    fix_H0=False,
    nwalkers=50,
    nsteps=1000,
    discard=200,
    output=False
)

# Distance modulus plot with residuals
plot_distance_modulus_with_residuals(z, mu, sigma_mu, samples_curved, flat_universe=False)

# Residual histograms 2x2
plot_residual_histograms(z, mu, sigma_mu, samples_flat, samples_curved, nsim=200)


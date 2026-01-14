import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from new_data_splitting import load_scp_data
from priors_posterior_curve import log_posterior_m  # new posterior that accepts Omega_L, Omega_k, L
from model_curve import m_model_curved  # curved model version

def run_supernova_mcmc_m(
    z,
    m,
    sigma_m,
    Omega_L_init=0.7,
    Omega_k_init=0.0,
    L_init=3e32,
    nwalkers=32,
    nsteps=100,
    discard=10,
    thin=1,
    plot_chains=True,
    plot_corner=True
):
    """
    Runs MCMC for supernova magnitude data including curvature.
    Fits Omega_Lambda, Omega_k, and L.
    """
    ndim = 3  # Omega_Lambda, Omega_k, L

    # Initialize walkers around initial guesses
    pos = np.zeros((nwalkers, ndim))
    pos[:, 0] = Omega_L_init + 1e-2 * np.random.randn(nwalkers)
    pos[:, 1] = Omega_k_init + 1e-3 * np.random.randn(nwalkers)
    pos[:, 2] = L_init * (1 + 1e-2 * np.random.randn(nwalkers))

    # Set up sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior_m,
        args=(z, m, sigma_m)
    )

    # Run MCMC
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Flatten chain
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # Compute acceptance fraction
    acceptance_frac = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction: {acceptance_frac:.3f}")
    print(f"Shape of samples: {samples.shape}")

    # Plot walker chains
    if plot_chains:
        fig, axes = plt.subplots(ndim, figsize=(10, 6), sharex=True)
        labels = [r"$\Omega_\Lambda$", r"$\Omega_k$", r"$L$"]
        for i in range(ndim):
            axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.3)
            axes[i].set_ylabel(labels[i])
        axes[-1].set_xlabel("Step")
        plt.show()

    # Corner plot
    if plot_corner:
        corner.corner(
            samples,
            labels=[r"$\Omega_\Lambda$", r"$\Omega_k$", r"$L$"],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g"
        )
        plt.show()

    # Compute parameter statistics
    param_stats = {}
    for i, name in enumerate(["Omega_Lambda", "Omega_k", "L"]):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        err_minus = p50 - p16
        err_plus = p84 - p50
        param_stats[name] = (p50, err_minus, err_plus)
        print(f"{name} = {p50:.4g} -{err_minus:.4g} +{err_plus:.4g}")

    return samples, acceptance_frac, param_stats


# ==========================
# Load data and run MCMC
# ==========================
from old_data import low_redshift_values, high_redshift_values

# Combine low and high redshift datasets
data_all = np.vstack([low_redshift_values, high_redshift_values])
z_m = data_all[:, 0]
m = data_all[:, 1]
sigma_m = data_all[:, 2]

# Run the MCMC
samples, acc, stats = run_supernova_mcmc_m(
    z_m,
    m,
    sigma_m,
    Omega_L_init=0.7,
    Omega_k_init=0.0,
    L_init=3e32,
    nwalkers=32,
    nsteps=100,
    discard=10,
    thin=1
)
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from Priors_posteriors import log_posterior
from new_data_splitting import z, mu, sigma_mu

def run_supernova_mcmc(
    z,
    mu,
    sigma_mu,
    Omega_L_init=0.7,
    Omega_k_init=0.0,
    H0_init=70.0,
    flat=False,           # True = flat universe, Omega_k fixed
    nwalkers=32,
    nsteps=500,
    discard=100,
    thin=1,
    plot_chains=True,
    plot_corner=True
):
    """
    Runs MCMC on supernova distance modulus data.
    Automatically handles flat (ndim=2) vs curved (ndim=3) universes.
    """

    # ------------------------
    # Set number of parameters
    # ------------------------
    if flat:
        ndim = 2  # Omega_L, H0
    else:
        ndim = 3  # Omega_L, Omega_k, H0

    # ------------------------
    # Initialize walkers
    # ------------------------
    pos = np.zeros((nwalkers, ndim))

    # Omega_L
    pos[:, 0] = Omega_L_init + 1e-2 * np.random.randn(nwalkers)

    # Omega_k if curved
    if not flat:
        pos[:, 1] = Omega_k_init + 1e-2 * np.random.randn(nwalkers)

    # H0
    pos[:, -1] = H0_init + 0.5 * np.random.randn(nwalkers) # larger spread

    # ------------------------
    # Wrap posterior to handle fixed Omega_k
    # ------------------------
    def posterior_wrap(theta, z, mu_obs, mu_err):
        if flat:
            # Only sample Omega_L and H0, set Omega_k = 0
            Omega_L, H0_val = theta
            Omega_k = 0.0
        else:
            Omega_L, Omega_k, H0_val = theta

        return log_posterior([Omega_L, Omega_k, H0_val], z, mu_obs, mu_err, flat=flat)

    # ------------------------
    # Set up sampler
    # ------------------------
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        posterior_wrap,
        args=(z, mu, sigma_mu)
    )

    # ------------------------
    # Run MCMC
    # ------------------------
    sampler.run_mcmc(pos, nsteps, progress=True)
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"Samples shape: {samples.shape}")

    # ------------------------
    # Trace plots
    # ------------------------
    if plot_chains:
        if flat:
            labels = [r"$\Omega_\Lambda$", r"$H_0$"]
        else:
            labels = [r"$\Omega_\Lambda$", r"$\Omega_k$", r"$H_0$"]

        fig, axes = plt.subplots(ndim, figsize=(10, 6), sharex=True)
        for i in range(ndim):
            axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.3)
            axes[i].set_ylabel(labels[i])
        axes[-1].set_xlabel("Step")
        plt.show()

    # ------------------------
    # Corner plot
    # ------------------------
    if plot_corner:
        corner.corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g"
        )
        plt.show()

    # ------------------------
    # Compute statistics
    # ------------------------
    stats = {}
    for i, name in enumerate(labels):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        stats[name] = (p50, p50 - p16, p84 - p50)
        print(f"{name} = {p50:.4g} -{p50-p16:.4g} +{p84-p50:.4g}")

    return samples, stats


#samples_flat, stats_flat = run_supernova_mcmc(z, mu, sigma_mu,flat=True,Omega_L_init=0.7,H0_init=70.0,nwalkers=32,nsteps=500)

samples_curve, stats_curve = run_supernova_mcmc(z, mu, sigma_mu,flat=False,Omega_L_init=0.7,Omega_k_init=0.0,H0_init=70.0,nwalkers=32,nsteps=500)
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
    flat=False,          # True → Omega_k fixed = 0
    fix_H0=False,        # True → H0 fixed
    H0_fixed=70.0,
    nwalkers=32,
    nsteps=200,
    discard=40,
    thin=1,
    plot_chains=True,
    plot_corner=True
):
    """
    MCMC runner for supernova cosmology.

    Supported cases:
    ----------------
    1D:  flat=True,  fix_H0=True        → ΩΛ
    2D:  flat=True,  fix_H0=False       → ΩΛ, H0
         flat=False, fix_H0=True        → ΩΛ, Ωk
    3D:  flat=False, fix_H0=False       → ΩΛ, Ωk, H0
    """

    # ------------------------
    # Determine dimensionality
    # ------------------------
    ndim = 1  # ΩΛ always sampled

    if not flat:
        ndim += 1  # Ωk

    if not fix_H0:
        ndim += 1  # H0

    # ------------------------
    # Initialise walkers
    # ------------------------
    pos = np.zeros((nwalkers, ndim))
    idx = 0

    # ΩΛ
    pos[:, idx] = Omega_L_init + 1e-2 * np.random.randn(nwalkers)
    idx += 1

    # Ωk
    if not flat:
        pos[:, idx] = Omega_k_init + 1e-2 * np.random.randn(nwalkers)
        idx += 1

    # H0
    if not fix_H0:
        pos[:, idx] = H0_init + 0.5 * np.random.randn(nwalkers)

    # ------------------------
    # Posterior wrapper
    # ------------------------
    def posterior_wrap(theta, z, mu_obs, mu_err):
        idx = 0

        Omega_L = theta[idx]
        idx += 1

        if not flat:
            Omega_k = theta[idx]
            idx += 1
        else:
            Omega_k = 0.0

        if not fix_H0:
            H0_val = theta[idx]
        else:
            H0_val = H0_fixed

        return log_posterior(
            [Omega_L, Omega_k, H0_val],
            z,
            mu_obs,
            mu_err,
            flat=flat
        )

    # ------------------------
    # Run sampler
    # ------------------------
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        posterior_wrap,
        args=(z, mu, sigma_mu)
    )

    sampler.run_mcmc(pos, nsteps, progress=True)
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"Samples shape: {samples.shape}")

    # ------------------------
    # Labels
    # ------------------------
    labels = [r"$\Omega_\Lambda$"]

    if not flat:
        labels.append(r"$\Omega_k$")

    if not fix_H0:
        labels.append(r"$H_0$")

    # ------------------------
    # Trace plots
    # ------------------------
    if plot_chains:
        fig, axes = plt.subplots(ndim, figsize=(10, 6), sharex=True)

        if ndim == 1:
            axes = [axes]

        for i in range(ndim):
            axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.3)
            axes[i].set_ylabel(labels[i])

        axes[-1].set_xlabel("Step")
        plt.show()

    # ------------------------
    # Corner plot
    # ------------------------
    if plot_corner and ndim > 1:
        corner.corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g"
        )
        plt.show()

    # ------------------------
    # Parameter summaries
    # ------------------------
    stats = {}
    for i, name in enumerate(labels):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        stats[name] = (p50, p50 - p16, p84 - p50)
        print(f"{name} = {p50:.4g} -{p50-p16:.4g} +{p84-p50:.4g}")

    return samples, stats

# 1D: ΩΛ only (flat, H0 fixed)
run_supernova_mcmc(z, mu, sigma_mu, flat=True, fix_H0=True)

# 2D: ΩΛ + H0 (flat)
run_supernova_mcmc(z, mu, sigma_mu, flat=True, fix_H0=False)

# 2D: ΩΛ + Ωk (H0 fixed)
#run_supernova_mcmc(z, mu, sigma_mu, flat=False, fix_H0=True)

# 3D: ΩΛ + Ωk + H0
run_supernova_mcmc(z, mu, sigma_mu, flat=False, fix_H0=False)

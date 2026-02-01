import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from Priors_posteriors import log_posterior
from new_data_splitting import z, mu, sigma_mu

def plot_1d_posterior(samples, label):
    """
    Plot a 1D posterior histogram with 16/50/84 percentiles.
    """
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])

    plt.figure(figsize=(4, 4))
    plt.hist(samples, bins=40, density=True, histtype='step', color='black')
    plt.axvline(p16, linestyle='--', color='black')
    plt.axvline(p50, linestyle='-', color='black')
    plt.axvline(p84, linestyle='--', color='black')

    plt.xlabel(label)
    plt.ylabel("Posterior density")
    plt.title(
        rf"{label} = {p50:.3g}$^{{+{p84-p50:.3g}}}_{{-{p50-p16:.3g}}}$"
    )
    plt.tight_layout()
    plt.show()

def run_supernova_mcmc(
    z,
    mu,
    sigma_mu,
    Omega_L_init=0.7,
    Omega_k_init=0.0,
    H0_init=70.0,
    prior_type="flat",      # <-- NEW
    flat_universe=False,    # <-- NEW
    fix_H0=False,
    H0_fixed=70.0,
    nwalkers=32,
    nsteps=200,
    discard=40,
    thin=1,
    output=True
):
    """
    MCMC runner for supernova cosmology.

    diagnostics=False → suppress ALL printing and plotting
    """

    # ------------------------
    # Dimensionality
    # ------------------------
    ndim = 1
    if not flat_universe:
        ndim += 1
    if not fix_H0:
        ndim += 1

    # ------------------------
    # Initial walker positions
    # ------------------------
    pos = np.zeros((nwalkers, ndim))
    idx = 0

    pos[:, idx] = Omega_L_init + 1e-2 * np.random.randn(nwalkers)
    idx += 1

    if not flat_universe:
        pos[:, idx] = Omega_k_init + 1e-2 * np.random.randn(nwalkers)
        idx += 1

    if not fix_H0:
        pos[:, idx] = H0_init + 0.5 * np.random.randn(nwalkers)

    # ------------------------
    # Posterior wrapper
    # ------------------------
    def posterior_wrap(theta, z, mu_obs, mu_err):
        idx = 0
        Omega_L = theta[idx]
        idx += 1

        Omega_k = theta[idx] if not flat_universe else 0.0
        if not flat_universe:
            idx += 1

        H0_val = theta[idx] if not fix_H0 else H0_fixed

        return log_posterior(
            [Omega_L, Omega_k, H0_val],
            z, mu_obs, mu_err,
            prior_type=prior_type,
            flat_universe=flat_universe
        )

    # ------------------------
    # Run sampler
    # ------------------------
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        posterior_wrap,
        args=(z, mu, sigma_mu)
    )

    sampler.run_mcmc(pos, nsteps, progress=output)

    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # ------------------------
    # Labels
    # ------------------------
    labels = [r"$\Omega_\Lambda$"]
    if not flat_universe:
        labels.append(r"$\Omega_k$")
    if not fix_H0:
        labels.append(r"$H_0$")

    # ------------------------
    # Diagnostics (ALL gated)
    # ------------------------
    if output:
        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(f"Samples shape: {samples.shape}")

        # Trace plots
        fig, axes = plt.subplots(ndim, figsize=(10, 6), sharex=True)
        if ndim == 1:
            axes = [axes]

        for i in range(ndim):
            axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.3)
            axes[i].set_ylabel(labels[i])

        axes[-1].set_xlabel("Step")
        plt.show()

        # Corner / 1D posterior
        if ndim == 1:
            plot_1d_posterior(samples[:, 0], labels[0])
        else:
            corner.corner(
                samples,
                labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt=".3g"
            )
            plt.show()

    # ------------------------
    # Parameter summaries (always returned, not printed)
    # ------------------------
    stats = {}
    for i, name in enumerate(labels):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        stats[name] = (p50, p50 - p16, p84 - p50)

        if output:
            print(f"{name} = {p50:.4g} -{p50-p16:.4g} +{p84-p50:.4g}")

    return samples, stats

# 1D: ΩΛ only (flat, H0 fixed)
run_supernova_mcmc(z, mu, sigma_mu, prior_type="flat", flat_universe=True, fix_H0=True, output=True)

# 2D: ΩΛ + H0 (flat)
#run_supernova_mcmc(z, mu, sigma_mu, prior_type="flat", flat_universe=True, fix_H0=False, output=True)

# 3D: ΩΛ + Ωk + H0
#run_supernova_mcmc(z, mu, sigma_mu, prior_type="flat", flat_universe=False, fix_H0=False, output=True)
#run_supernova_mcmc(z, mu, sigma_mu, prior_type="gaussian", flat_universe=False, fix_H0=False, output=True)

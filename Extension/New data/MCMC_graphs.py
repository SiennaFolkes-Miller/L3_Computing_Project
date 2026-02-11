import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from Priors_posteriors import log_posterior


def plot_1d_posterior(samples, label):
    """
    Plot a 1D posterior histogram with 16/50/84 percentiles.
    Publication-quality formatting (no bold text).
    """
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])

    plt.figure(figsize=(5.5, 4.5))

    # Histogram
    plt.hist(
        samples,
        bins=40,
        density=True,
        histtype='step',
        linewidth=2,
        color='black'
    )

    # Percentile lines with legend labels
    plt.axvline(p16, linestyle='--', linewidth=2, color='black', label='16%')
    plt.axvline(p50, linestyle='--',  linewidth=2.5, color='black', label='median')
    plt.axvline(p84, linestyle='--', linewidth=2, color='black', label='84%')

    # Axis labels (larger, not bold)
    plt.xlabel(label, fontsize=18)
    plt.ylabel("Posterior density", fontsize=18)

    # Larger tick labels
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Larger title (not bold)
    plt.title(
        rf"{label} = {p50:.3g}$^{{+{p84-p50:.3g}}}_{{-{p50-p16:.3g}}}$",
        fontsize=17
    )

    # Legend
    #plt.legend(loc = 'upper right', fontsize=12)

    plt.tight_layout()
    plt.show()


def run_supernova_mcmc(
    z,
    mu,
    sigma_mu,
    Omega_L_init=0.7,
    Omega_k_init=0.0,
    H0_init=70.0,
    prior_type="flat_wide",
    flat_universe=False,
    fix_H0=False,
    H0_fixed=70.0,
    nwalkers=32,
    nsteps=500,
    discard=None,
    thin=None,
    output=True
):
    """
    MCMC runner for supernova cosmology with autocorrelation diagnostics.
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

    pos[:, idx] = Omega_L_init + 0.05 * np.random.randn(nwalkers)
    idx += 1

    if not flat_universe:
        pos[:, idx] = Omega_k_init + 0.05 * np.random.randn(nwalkers)
        idx += 1

    if not fix_H0:
        pos[:, idx] = H0_init + 1.0 * np.random.randn(nwalkers)

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

    # ------------------------
    # Autocorrelation diagnostics
    # ------------------------
    try:
        tau = sampler.get_autocorr_time()
        if output:
            print("\nAutocorrelation times (τ):", tau)
        burn_in = int(2 * np.max(tau)) if discard is None else discard
        thin_by = int(0.5 * np.min(tau)) if thin is None else thin
    except emcee.autocorr.AutocorrError:
        burn_in = discard if discard is not None else 0
        thin_by = thin if thin is not None else 1

    # ------------------------
    # Flatten chain
    # ------------------------
    samples = sampler.get_chain(discard=burn_in, thin=thin_by, flat=True)

    # ------------------------
    # Labels
    # ------------------------
    labels = [r"$\Omega_\Lambda$"]
    if not flat_universe:
        labels.append(r"$\Omega_k$")
    if not fix_H0:
        labels.append(r"$H_0$")

    # ------------------------
    # Diagnostics
    # ------------------------
    if output:
        print(f"\nMean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(f"Flat samples shape: {samples.shape}")

        # Trace plots
        fig, axes = plt.subplots(ndim, figsize=(10, 4), sharex=True)
        if ndim == 1:
            axes = [axes]

        chain = sampler.get_chain()

        for i in range(ndim):
            axes[i].plot(chain[:, :, i], alpha=0.35, linewidth=1)
            axes[i].set_ylabel(labels[i], fontsize=18)
            axes[i].tick_params(axis='both', labelsize=15)
            axes[i].set_xlim(0, chain.shape[0] - 1)

        axes[-1].set_xlabel("Step", fontsize=18)
        plt.tight_layout()
        plt.show()

        # Posterior plot
        if ndim == 1:
            plot_1d_posterior(samples[:, 0], labels[0])
        else:
            corner.corner(
                samples,
                labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt=".3g",
                label_kwargs={"fontsize": 18},
                title_kwargs={"fontsize": 16},
                tick_kwargs={"labelsize": 14}
            )
            plt.show()

    # ------------------------
    # Covariance / variance
    # ------------------------
    if ndim >= 2:
        cov_matrix = np.cov(samples.T)
        eigvals, _ = np.linalg.eigh(cov_matrix)
        cond_number = np.max(eigvals) / np.min(eigvals)
    else:
        cov_matrix = np.var(samples, ddof=1)
        eigvals = np.array([cov_matrix])
        cond_number = np.nan

    # ------------------------
    # Parameter summaries
    # ------------------------
    stats = {}
    for i, name in enumerate(labels):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        stats[name] = (p50, p50 - p16, p84 - p50)
        if output:
            print(f"{name} = {p50:.4g} -{p50-p16:.4g} +{p84-p50:.4g}")

    return samples, stats


from new_data_splitting import z, mu, sigma_mu

run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_narrow", flat_universe=True, fix_H0=True, H0_fixed = 70.0, nwalkers=50, nsteps=2000,discard=400,thin=1,output=True)
#run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_narrow", flat_universe=True, fix_H0=False, H0_fixed = 70.0, nwalkers=75, nsteps=3000,discard=600,thin=1,output=True)
#run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_narrow", flat_universe=False, fix_H0=False, H0_fixed = 70.0, nwalkers=100, nsteps=5000,discard=1000,thin=1,output=True)

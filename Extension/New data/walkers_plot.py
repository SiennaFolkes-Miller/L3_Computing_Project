import numpy as np
import emcee
import matplotlib.pyplot as plt
from Priors_posteriors import log_posterior
from new_data_splitting import z, mu, sigma_mu  # your supernova data


def run_1d_mcmc(
    z, mu, sigma_mu,
    Omega_L_init=0.7,
    nwalkers=50,
    nsteps=2000,
    burn_in=400,
    thin=1,
    prior_type="flat_wide",
    output=True
):
    """
    Runs 1D MCMC for Omega_Lambda.
    Returns: chain (nsteps, nwalkers), flattened samples, tau, burn_in
    """
    ndim = 1
    pos = Omega_L_init + 0.05 * np.random.randn(nwalkers, ndim)

    def posterior(theta, z, mu_obs, mu_err):
        Omega_L = theta[0]
        return log_posterior([Omega_L, 0.0, 70.0], z, mu_obs, mu_err,
                             prior_type=prior_type, flat_universe=True)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(z, mu, sigma_mu))
    sampler.run_mcmc(pos, nsteps, progress=output)

    # Autocorrelation
    try:
        tau = sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError:
        tau = np.array([np.nan])

    # Chain and flattened samples
    chain = sampler.get_chain()[:, :, 0]  # (nsteps, nwalkers)
    samples_flat = chain[burn_in::thin, :].flatten()

    if output:
        print(f"Autocorrelation time (tau): {tau}")
        print(f"Using fixed burn-in steps: {burn_in}")
        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    return chain, samples_flat, tau, burn_in


def plot_walkers_hist_final(chain, samples_flat, tau=None, burn_in=400, label=r"$\Omega_\Lambda$"):
    """
    1D walker + histogram plot with:
    - touching panels
    - shared y-axis
    - no histogram outlines
    - burn-in shading
    - horizontal arrow for tau
    - legend for median, percentiles, tau
    """
    nsteps, nwalkers = chain.shape
    # Percentiles
    p16, p50, p84 = np.percentile(samples_flat, [16, 50, 84])

    # -------------------
    # Figure and axes
    # -------------------
    fig, (ax_walk, ax_hist) = plt.subplots(
        1, 2, figsize=(12,5),
        gridspec_kw={'width_ratios':[3,1], 'wspace':0},  # no gap
        sharey=True
    )

    # ----- Walker plot -----
    for w in range(nwalkers):
        ax_walk.plot(np.arange(nsteps), chain[:, w], alpha=0.4, color='C0', linewidth=1)

    # Burn-in shaded
    ax_walk.axvspan(0, burn_in, color='gray', alpha=0.3, label='Burn-in')

    # Autocorrelation arrow at top
    if tau is not None and not np.isnan(tau[0]):
        y_top = ax_walk.get_ylim()[1]
        ax_walk.annotate(
            '', xy=(tau[0], y_top*0.95), xytext=(0, y_top*0.95),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2)
        )
        # Include tau in legend instead of text
        tau_legend_label = f"τ ≈ {tau[0]:.1f}"
    else:
        tau_legend_label = None

    ax_walk.set_xlabel("Step", fontsize=14)
    ax_walk.set_ylabel(label, fontsize=14)
    ax_walk.tick_params(axis='both', labelsize=12)
    ax_walk.set_xlim(0, nsteps)

    # ----- Histogram -----
    hist, bins = np.histogram(samples_flat, bins=40, density=True)
    bin_centers = 0.5*(bins[:-1] + bins[1:])
    max_width = nsteps * 0.8
    hist_scaled = hist / np.max(hist) * max_width

    # Histogram bars without edge color
    ax_hist.barh(bin_centers, hist_scaled, height=bins[1]-bins[0], left=0,
                 color='C0', edgecolor=None, linewidth=0)

    # Percentile lines
    ax_hist.axhline(p16, linestyle='--', color='orange', linewidth=2)
    ax_hist.axhline(p50, linestyle='-', color='red', linewidth=3)
    ax_hist.axhline(p84, linestyle='--', color='orange', linewidth=2)

    ax_hist.set_xlim(0, max_width)
    ax_hist.set_xlabel("Density (scaled)", fontsize=14)
    ax_hist.set_yticks([])  # no y-ticks
    ax_hist.set_ylabel("")   # no y-label
    ax_hist.set_xticks([500, 1000, 1500])

    # -------------------
    # Legend
    # -------------------
    handles = []
    labels = []

    # Burn-in
    handles.append(plt.Line2D([0], [0], color='gray', lw=4, alpha=0.3))
    labels.append('Burn-in')

    # Percentiles
    handles.append(plt.Line2D([0], [0], color='red', lw=3))
    labels.append('Median')
    handles.append(plt.Line2D([0], [0], color='orange', lw=2, linestyle='--'))
    labels.append('16-84 percentiles')

    # Tau
    if tau_legend_label:
        handles.append(plt.Line2D([0], [0], color='green', lw=2))
        labels.append(tau_legend_label)

    ax_walk.legend(handles, labels, fontsize=10)

    plt.tight_layout()
    plt.show()


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    chain, samples_flat, tau, burn_in = run_1d_mcmc(
        z, mu, sigma_mu,
        Omega_L_init=0.7,
        nwalkers=50,
        nsteps=2000,
        burn_in=400,
        thin=1,
        output=True
    )

    plot_walkers_hist_final(chain, samples_flat, tau=tau, burn_in=burn_in, label=r"$\Omega_\Lambda$")
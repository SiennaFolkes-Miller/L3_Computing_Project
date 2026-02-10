# redshift_scan.py
import numpy as np
from new_data_splitting import z, mu, sigma_mu
from MCMC import run_supernova_mcmc  # rename if needed

def apply_redshift_cut(z, mu, sigma_mu, zmin=0.0, zmax=None):
    mask = np.ones_like(z, dtype=bool)
    if zmin is not None:
        mask &= z >= zmin
    if zmax is not None:
        mask &= z <= zmax
    return z[mask], mu[mask], sigma_mu[mask]


def run_redshift_scan(
    zmax_list,
    prior_type="flat_wide",
    flat_universe=False,
    fix_H0=False,
    output=True
):
    """
    Run MCMC for increasing maximum redshift cuts.
    """
    results = {}

    for zmax in zmax_list:
        z_sel, mu_sel, sig_sel = apply_redshift_cut(
            z, mu, sigma_mu,
            zmin=0.0,
            zmax=zmax
        )

        if output:
            print(f"\n=== z ≤ {zmax:.2f} | N = {len(z_sel)} ===")

        samples, stats = run_supernova_mcmc(
            z_sel,
            mu_sel,
            sig_sel,
            prior_type=prior_type,
            flat_universe=flat_universe,
            fix_H0=fix_H0, nwalkers=50, nsteps=3000,discard=600,thin=1,
            output=output
        )

        results[zmax] = {
            "samples": samples,
            "stats": stats,
            "N": len(z_sel)
        }

    return results

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_redshift_scan_omegaL_omegaK(results, zmax_values, bins=30):
    """
    For each zmax:
      - ΩΛ marginal histogram (top)
      - ΩΛ–Ωk contour (bottom)
    """

    for zmax in zmax_values:
        samples = results[zmax]["samples"]

        # Extract parameters (curved universe assumed)
        Omega_L = samples[:, 0]
        Omega_k = samples[:, 1]

        # --- KDE for ΩΛ–Ωk ---
        kde = gaussian_kde([Omega_k, Omega_L])

        x = np.linspace(Omega_k.min(), Omega_k.max(), 200)
        y = np.linspace(Omega_L.min(), Omega_L.max(), 200)
        X, Y = np.meshgrid(x, y)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        # --- Figure layout (corner-style) ---
        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)

        ax_hist = fig.add_subplot(gs[0])
        ax_cont = fig.add_subplot(gs[1], sharex=ax_hist)

        # --- ΩΛ histogram ---
        ax_hist.hist(
            Omega_L,
            bins=bins,
            density=True,
            color="#A6CEE3",   # light blue
            alpha=0.8
        )
        ax_hist.set_ylabel("Density")
        ax_hist.tick_params(labelbottom=False)

        # --- ΩΛ–Ωk contour ---
        ax_cont.contour(
            X, Y, Z,
            levels=5,
            colors="black"
        )
        ax_cont.set_xlabel(r"$\Omega_k$")
        ax_cont.set_ylabel(r"$\Omega_\Lambda$")

        # Reference lines (optional but standard)
        ax_cont.axvline(0.0, ls="--", lw=1, color="gray", alpha=0.6)
        ax_cont.axhline(0.7, ls="--", lw=1, color="gray", alpha=0.6)

        # Clean look for report
        ax_hist.set_title(rf"$z \le {zmax:.2f}$", fontsize=12)
        for ax in [ax_hist, ax_cont]:
            ax.grid(False)

        plt.show()
zmax_values = np.linspace(0.1, np.max(z), 6)

results = run_redshift_scan(
    zmax_values,
    prior_type="flat_wide",
    flat_universe=False,
    fix_H0=False,
    output=False
)

plot_redshift_scan_omegaL_omegaK(results, zmax_values)

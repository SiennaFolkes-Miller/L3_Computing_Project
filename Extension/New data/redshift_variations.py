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

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 14

import corner
import matplotlib.pyplot as plt

def plot_corner_consistent(
        samples,
        labels,
        truths=None,
        title=None,
        color="blue"):

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": AXIS_LABEL_SIZE},
        label_kwargs={"fontsize": AXIS_LABEL_SIZE},
        color=color,
        hist_kwargs={"linewidth": 2},
        contour_kwargs={"linewidths": 2}
    )

    # --------------------------------------------------
    # Enforce tick label sizes everywhere
    # --------------------------------------------------
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)

    # Optional overall title
    if title is not None:
        fig.suptitle(title, fontsize=AXIS_LABEL_SIZE)

    plt.tight_layout()
    return fig

zmax_values = np.linspace(0.1, np.max(z), 6)

results = run_redshift_scan(
    zmax_values,
    prior_type="flat_wide",
    flat_universe=False,
    fix_H0=False
)

# --------------------------------------------------
# Loop over all cuts and plot
# --------------------------------------------------

for zmax in sorted(results.keys()):

    samples = results[zmax]["samples"]

    # Parameter labels
    if samples.shape[1] == 2:
        labels = [
            r"$\Omega_\Lambda$",
            r"$H_0$"
        ]
    elif samples.shape[1] == 3:
        labels = [
            r"$\Omega_\Lambda$",
            r"$\Omega_k$",
            r"$H_0$"
        ]
    else:
        raise ValueError("Unexpected parameter dimension")

    fig = plot_corner_consistent(
        samples,
        labels=labels,
        title=f"Posterior for $z \\leq {zmax:.2f}$"
    )

    plt.show()
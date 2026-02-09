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


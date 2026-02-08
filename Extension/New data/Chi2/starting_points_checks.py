import numpy as np
import matplotlib.pyplot as plt

from MCMC import run_supernova_mcmc
from Chi2 import run_supernova_chi2
from new_data_splitting import z, mu, sigma_mu

# -------------------------------------------------
# Define different starting points to test
# -------------------------------------------------
STARTING_POINTS = [
    dict(Omega_L=0.6, Omega_k=0.0, H0=65.0),
    dict(Omega_L=0.7, Omega_k=0.0, H0=70.0),
    dict(Omega_L=0.8, Omega_k=0.0, H0=75.0),
    dict(Omega_L=0.9, Omega_k=0.1, H0=68.0),
]

# -------------------------------------------------
# Run χ² and MCMC for a single starting point
# -------------------------------------------------
def run_single_start(z, mu, sigma_mu, start,
                     flat_universe=False,
                     fix_H0=False,
                     prior_type="flat"):
    """
    Run χ² and MCMC from a single starting point.
    """

    # χ²
    bf_chi2, _, chi2_errs = run_supernova_chi2(
        z, mu, sigma_mu,
        Omega_L_init=start["Omega_L"],
        Omega_k_init=start["Omega_k"],
        H0_init=start["H0"],
        flat_universe=flat_universe,
        fix_H0=fix_H0,
        output=False
    )

    # MCMC
    samples, stats = run_supernova_mcmc(
        z, mu, sigma_mu,
        Omega_L_init=start["Omega_L"],
        Omega_k_init=start["Omega_k"],
        H0_init=start["H0"],
        flat_universe=flat_universe,
        fix_H0=fix_H0,
        prior_type=prior_type,
        output=False
    )

    return bf_chi2, chi2_errs, stats

# -------------------------------------------------
# Main sensitivity analysis
# -------------------------------------------------
def starting_value_sensitivity(z, mu, sigma_mu,
                               flat_universe=False,
                               fix_H0=False,
                               prior_type="flat"):
    """
    Test sensitivity of χ² and MCMC results to starting values.
    """

    print("\n===================================")
    print("Starting value sensitivity analysis")
    print("===================================")
    print(f"Flat universe: {flat_universe}, Fix H0: {fix_H0}")
    print(f"Prior: {prior_type}")

    results = []

    for i, start in enumerate(STARTING_POINTS):
        print(f"\n--- Start {i+1} ---")
        print(start)

        bf_chi2, chi2_errs, stats = run_single_start(
            z, mu, sigma_mu, start,
            flat_universe=flat_universe,
            fix_H0=fix_H0,
            prior_type=prior_type
        )

        print("Chi² best fit:")
        for val, err, name in zip(bf_chi2, chi2_errs,
                                  stats.keys()):
            print(f"  {name} = {val:.4g} -{err[0]:.4g} +{err[1]:.4g}")

        print("MCMC median ±16/84%:")
        for k, (med, minus, plus) in stats.items():
            print(f"  {k} = {med:.4g} -{minus:.4g} +{plus:.4g}")

        results.append((bf_chi2, stats))

    return results

# -------------------------------------------------
# Example usage
# -------------------------------------------------
# 1D: flat universe, fixed H0
results_1d = starting_value_sensitivity(
    z, mu, sigma_mu,
    flat_universe=True,
    fix_H0=True,
    prior_type="flat"
)

# 2D: flat universe, free H0
results_2d = starting_value_sensitivity(
    z, mu, sigma_mu,
    flat_universe=True,
    fix_H0=False,
    prior_type="flat"
)

# 3D: curved universe
results_3d = starting_value_sensitivity(
    z, mu, sigma_mu,
    flat_universe=False,
    fix_H0=False,
    prior_type="flat"
)

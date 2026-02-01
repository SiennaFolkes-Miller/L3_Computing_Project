import numpy as np
from Chi2 import run_supernova_chi2
from MCMC import run_supernova_mcmc
from new_data_splitting import z, mu, sigma_mu
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Model configuration
# -----------------------------
def configure_model(case):
    """
    case:
      '1D' → ΩΛ only (flat, H0 fixed)
      '2D' → ΩΛ, H0 (flat)
      '3D' → ΩΛ, Ωk, H0
    """
    if case == "1D":
        return dict(
            flat=True,
            fix_H0=True,
            labels=[r"$\Omega_\Lambda$"]
        )
    if case == "2D":
        return dict(
            flat=True,
            fix_H0=False,
            labels=[r"$\Omega_\Lambda$", r"$H_0$"]
        )
    if case == "3D":
        return dict(
            flat=False,
            fix_H0=False,
            labels=[r"$\Omega_\Lambda$", r"$\Omega_k$", r"$H_0$"]
        )
    raise ValueError("case must be '1D', '2D', or '3D'")


# -----------------------------
# Run LOO with optional subsampling
# -----------------------------
def run_loo(z, mu, sigma_mu, *, flat, fix_H0, n_samples=None, strategy="random"):
    """
    Run leave-one-out sensitivity analysis.
    If n_samples is None, loops over all points.
    Otherwise, subsamples n_samples indices.
    strategy: 'random' or 'redshift'
    """
    n_points = len(z)
    if n_samples is None or n_samples >= n_points:
        loo_indices = np.arange(n_points)
    elif strategy == "random":
        loo_indices = np.random.choice(n_points, size=n_samples, replace=False)
    elif strategy == "redshift":
        bins = min(n_samples, 10)
        bin_edges = np.linspace(z.min(), z.max(), bins + 1)
        loo_indices = []
        for i in range(bins):
            mask = (z >= bin_edges[i]) & (z < bin_edges[i+1])
            idx = np.where(mask)[0]
            if len(idx) > 0:
                loo_indices.append(np.random.choice(idx))
        loo_indices = np.array(loo_indices)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    results_chi2 = []
    results_mcmc = []
        # ------------------
        # χ²
        # ------------------
    true_chi2, _, _ = run_supernova_chi2(z, mu, sigma_mu,flat=flat, fix_H0=fix_H0,output=False)
    _, true_mcmc_stats = run_supernova_mcmc(z, mu, sigma_mu,flat=flat, fix_H0=fix_H0,output=False)
    for i in loo_indices:
        z_loo = np.delete(z, i)
        mu_loo = np.delete(mu, i)
        sigma_loo = np.delete(sigma_mu, i)

        # ------------------
        # χ²
        # ------------------
        bf_chi2, _, _ = run_supernova_chi2(
            z_loo, mu_loo, sigma_loo,
            flat=flat, fix_H0=fix_H0,
            output=False
        )

        # ------------------
        # MCMC
        # ------------------
        samples, stats = run_supernova_mcmc(
            z_loo, mu_loo, sigma_loo,
            flat=flat, fix_H0=fix_H0,
            output=False
        )

        results_chi2.append(bf_chi2)
        results_mcmc.append([stats[k][0] for k in stats])

    return np.array(results_chi2), np.array(results_mcmc), true_chi2, true_mcmc_stats


# -----------------------------
# Plotting function
# -----------------------------
def plot_loo_distribution(values, label, method_name, true_value=None, bins=30):
    """
    Plot leave-one-out (LOO) distribution with mean, median, and optional true dataset value.

    Parameters
    ----------
    values : array-like
        LOO values for the parameter.
    label : str
        Parameter label for x-axis.
    method_name : str
        "Chi-squared" or "MCMC".
    true_value : float, optional
        Full-dataset best-fit value to overlay.
    bins : int
        Number of histogram bins.
    """
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values, ddof=1)
    stderr = std / np.sqrt(len(values))

    plt.figure(figsize=(5, 4))
    
    # Histogram
    plt.hist(values, bins=bins, density=True, histtype='stepfilled',
             alpha=0.3, color='gray', edgecolor='black', linewidth=1.2, label='LOO samples')
    
    # Mean
    plt.axvline(mean, linestyle='--', color='blue', linewidth=2, label=f"Mean = {mean:.4g}")
    
    # Median
    plt.axvline(median, linestyle='-', color='green', linewidth=2, label=f"Median = {median:.4g}")
    
    # True value (if given)
    if true_value is not None:
        plt.axvline(true_value, linestyle=':', color='red', linewidth=2, label=f"Full dataset = {true_value:.4g}")
    
    plt.xlabel(label)
    plt.ylabel("Density")
    plt.title(f"{method_name} – Leave-one-out")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print stats
    print(f"{method_name} – {label}")
    print(f"  Mean     = {mean:.5g}")
    print(f"  Median   = {median:.5g}")
    print(f"  Std dev  = {std:.5g}")
    print(f"  Std err  = {stderr:.5g}")
    if true_value is not None:
        print(f"  True val = {true_value:.5g}")


# -----------------------------
# Main wrapper
# -----------------------------
def loo_sensitivity_analysis(z, mu, sigma_mu, case="1D", n_samples=None, strategy="random"):
    """
    Run LOO sensitivity for a given case (1D, 2D, 3D) with optional subsampling.

    Parameters
    ----------
    z, mu, sigma_mu : array-like
        Supernova dataset.
    case : str
        '1D', '2D', or '3D' → which parameters to vary.
    n_samples : int, optional
        Number of LOO points to sample (if None, uses all points).
    strategy : str
        Sampling strategy if n_samples is set. Currently supports "random".
    """

    cfg = configure_model(case)

    # ------------------------
    # Compute full dataset true values
    # ------------------------
    true_chi2, _, _ = run_supernova_chi2(
        z, mu, sigma_mu,
        flat=cfg["flat"], fix_H0=cfg["fix_H0"],
        output=False
    )

    true_samples, true_stats = run_supernova_mcmc(
        z, mu, sigma_mu,
        flat=cfg["flat"], fix_H0=cfg["fix_H0"],
        output=False
    )

    # ------------------------
    # Run LOO (with optional subsampling)
    # ------------------------
    if n_samples is not None and n_samples < len(z):
        if strategy == "random":
            loo_indices = np.random.choice(len(z), size=n_samples, replace=False)
        else:
            raise ValueError("Only 'random' strategy is supported currently.")
    else:
        loo_indices = np.arange(len(z))

    results_chi2 = []
    results_mcmc = []

    for i in loo_indices:
        z_loo = np.delete(z, i)
        mu_loo = np.delete(mu, i)
        sigma_loo = np.delete(sigma_mu, i)

        # ------------------
        # χ²
        # ------------------
        bf_chi2, _, _ = run_supernova_chi2(
            z_loo, mu_loo, sigma_loo,
            flat=cfg["flat"], fix_H0=cfg["fix_H0"],
            output=False
        )

        # ------------------
        # MCMC
        # ------------------
        samples, stats = run_supernova_mcmc(
            z_loo, mu_loo, sigma_loo,
            flat=cfg["flat"], fix_H0=cfg["fix_H0"],
            output=False
        )

        results_chi2.append(bf_chi2)
        results_mcmc.append([stats[k][0] for k in stats])

    results_chi2 = np.array(results_chi2)
    results_mcmc = np.array(results_mcmc)

    # ------------------------
    # Plot distributions with true values
    # ------------------------
    for i, label in enumerate(cfg["labels"]):
        plot_loo_distribution(
            results_chi2[:, i],
            label,
            "Chi-squared",
            true_value=true_chi2[i]
        )

        plot_loo_distribution(
            results_mcmc[:, i],
            label,
            "MCMC",
            true_value=true_stats[cfg["labels"][i]][0]  # median from full dataset
        )

    return results_chi2, results_mcmc, true_chi2, true_stats


# -----------------------------
# Example usage
# -----------------------------
# For large datasets (~750 points), sample 100 points for LOO
# 1D: flat + H0 fixed
#results_chi2_1d, results_mcmc_1d, _, _ = loo_sensitivity_analysis(z, mu, sigma_mu, case="1D", n_samples=10)

# 2D: flat
results_chi2_2d, results_mcmc_2d, _, _ = loo_sensitivity_analysis(z, mu, sigma_mu, case="2D", n_samples=10)

# 3D: curved
#results_chi2_3d, results_mcmc_3d, _, _ = loo_sensitivity_analysis(z, mu, sigma_mu, case="3D", n_samples=10)

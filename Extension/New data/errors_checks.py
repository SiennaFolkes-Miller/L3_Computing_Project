import numpy as np
import matplotlib.pyplot as plt
from MCMC import run_supernova_mcmc
from Chi2 import run_supernova_chi2
from new_data_splitting import z, mu, sigma_mu

# -----------------------------
# Utility to modify errors
# -----------------------------
def inflate_errors(sigma, factor):
    """Multiply all distance modulus errors by a constant factor."""
    return sigma * factor

def add_intrinsic_scatter(sigma, sigma_int):
    """Add intrinsic scatter in quadrature to distance modulus errors."""
    return np.sqrt(sigma**2 + sigma_int**2)

# -----------------------------
# Run both χ² and MCMC for a given error set
# -----------------------------
def run_analysis(z, mu, sigma, prior_type="flat", flat_universe=True, fix_H0=False):
    bf_chi2, _, chi2_errs = run_supernova_chi2(z, mu, sigma,
                                                flat_universe=flat_universe,
                                                fix_H0=fix_H0,
                                                output=False)
    samples, stats = run_supernova_mcmc(z, mu, sigma,
                                        flat_universe=flat_universe,
                                        fix_H0=fix_H0,
                                        prior_type=prior_type,
                                        output=False)
    return bf_chi2, chi2_errs, samples, stats

# -----------------------------
# Flexible error analysis
# -----------------------------
def run_error_analysis(z, mu, sigma_mu,
                       flat_universe=True,
                       fix_H0=False,
                       prior_type="flat",
                       inflate_factor=None,
                       intrinsic_scatter=None):
    """
    Run χ² and MCMC with modified distance modulus errors.
    """
    # Copy original errors
    sigma = sigma_mu.copy()

    # Modify errors
    if inflate_factor is not None:
        sigma = inflate_errors(sigma, inflate_factor)
    if intrinsic_scatter is not None:
        sigma = add_intrinsic_scatter(sigma, intrinsic_scatter)

    # Run analysis
    bf_chi2, chi2_errs, samples, mcmc_stats = run_analysis(
        z, mu, sigma,
        flat_universe=flat_universe,
        fix_H0=fix_H0
    )

    # Convert χ² output to same format as MCMC stats
    labels = list(mcmc_stats.keys())
    chi2_stats = {}
    for i, lbl in enumerate(labels):
        val = bf_chi2[i]
        minus, plus = chi2_errs[i]
        chi2_stats[lbl] = (val, minus, plus)

    return chi2_stats, mcmc_stats, samples


# -----------------------------
# Optional plotting function for comparing error treatments
# -----------------------------
def compare_error_treatments(z, mu, sigma_mu,
                             flat_universe=True,
                             fix_H0=False,
                             prior_type="flat"):
    scenarios = {
        "Original": dict(),
        "Inflated x1.5": dict(inflate_factor=1.5),
        "Intrinsic 0.1 mag": dict(intrinsic_scatter=0.1)
    }

    scenario_colors = {
        "Original": "blue",
        "Inflated x1.5": "orange",
        "Intrinsic 0.1 mag": "green"
    }

    results = {}

    for name, kwargs in scenarios.items():
        print(f"\n--- {name} ---")

        chi2_stats, mcmc_stats, samples = run_error_analysis(
            z, mu, sigma_mu,
            flat_universe=flat_universe,
            fix_H0=fix_H0,
            prior_type=prior_type,
            **kwargs
        )

        results[name] = {
            "chi2": chi2_stats,
            "mcmc": mcmc_stats
        }

        # Pretty print (optional)
        print("Chi² best fit ±16/84%:")
        for k, (v, m, p) in chi2_stats.items():
            print(f"  {k} = {v:.4g} -{m:.4g} +{p:.4g}")

        print("MCMC median ±16/84%:")
        for k, (v, m, p) in mcmc_stats.items():
            print(f"  {k} = {v:.4g} -{m:.4g} +{p:.4g}")

    # -----------------------------
    # Plot ratios (shared function)
    # -----------------------------
    def plot_ratios(method):
        labels = list(results["Original"][method].keys())
        originals = results["Original"][method]

        plt.figure(figsize=(6,4))

        for i, lbl in enumerate(labels):
            ref = originals[lbl][0]

            for j, name in enumerate(scenarios):
                val, minus, plus = results[name][method][lbl]

                plt.errorbar(
                    i + j*0.2,
                    val / ref,
                    yerr=[[minus / ref], [plus / ref]],
                    fmt='o',
                    color=scenario_colors[name],
                    label=name if i == 0 else "",
                    capsize=3
                )

        plt.xticks(range(len(labels)), labels)
        plt.axhline(1.0, color='gray', linestyle='--')
        plt.ylabel("Parameter / Original value")
        plt.title(f"{method.upper()} comparison (ratios)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_ratios("mcmc")
    plot_ratios("chi2")

    return results


# -----------------------------
# Example usage
# -----------------------------
results_1d = compare_error_treatments(z, mu, sigma_mu,flat_universe=True,fix_H0=True,prior_type="flat")

#results_2d = compare_error_treatments(z, mu, sigma_mu,flat_universe=True,fix_H0=False,prior_type="flat")

results_3d = compare_error_treatments(z, mu, sigma_mu,flat_universe=False,fix_H0=False,prior_type="flat")


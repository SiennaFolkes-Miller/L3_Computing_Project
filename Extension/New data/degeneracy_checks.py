import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from new_data_splitting import z, mu, sigma_mu
from MCMC import run_supernova_mcmc
from Chi2 import run_supernova_chi2

# -----------------------------
# Utility: 2D ellipse from covariance
# -----------------------------
def plot_cov_ellipse(cov, mean, ax, n_std=1.0, facecolor='none', edgecolor='red', **kwargs):
    """
    Plot a 2D ellipse corresponding to covariance matrix.
    n_std: number of standard deviations (1 = 68%)
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=angle,
                    facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    ax.add_patch(ellip)

# -----------------------------
# Compute covariance matrix from samples or χ² points
# -----------------------------
def compute_covariance_matrix(samples):
    """
    samples: N x M array of parameter samples
    Returns covariance matrix (M x M)
    """
    return np.cov(samples.T)

# -----------------------------
# Compute degeneracy metrics
# -----------------------------
def degeneracy_metrics(samples, labels=None):
    """
    Compute correlations, eigenvalues, and condition numbers.
    """
    cov = compute_covariance_matrix(samples)
    corr = np.corrcoef(samples.T)
    eigvals = np.linalg.eigvals(cov)
    cond_number = eigvals.max() / eigvals.min() if eigvals.min() > 0 else np.inf

    if labels is None:
        labels = [f"p{i}" for i in range(samples.shape[1])]

    metrics = {
        'covariance': cov,
        'correlation': corr,
        'eigenvalues': eigvals,
        'condition_number': cond_number,
        'labels': labels
    }
    return metrics

# -----------------------------
# Plot 2D degeneracy with ellipse
# -----------------------------
def plot_2d_degeneracy(samples, idx_x=0, idx_y=1, label_x=None, label_y=None,
                       ax=None, n_std=1.0, method_name="MCMC", color='blue'):
    """
    Scatter plot of two parameters with 1-sigma covariance ellipse
    """
    x = samples[:, idx_x]
    y = samples[:, idx_y]
    mean = [np.mean(x), np.mean(y)]
    cov = np.cov(x, y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    ax.scatter(x, y, s=10, alpha=0.3, label=f"{method_name} samples", color=color)
    plot_cov_ellipse(cov, mean, ax, n_std=n_std, edgecolor=color)
    ax.set_xlabel(label_x if label_x else f"p{idx_x}")
    ax.set_ylabel(label_y if label_y else f"p{idx_y}")
    ax.set_title(f"{method_name} – 2D degeneracy")
    ax.legend()
    plt.tight_layout()
    if ax is None:
        plt.show()

# -----------------------------
# Main function: compare degeneracies
# -----------------------------
def compare_degeneracy(z, mu, sigma_mu, case="2D", output=True):
    """
    Run inference with MCMC and Chi2, return degeneracy metrics
    case: '1D', '2D', '3D'
    """
    # Determine dimensionality and labels
    if case == "1D":
        flat=True; fix_H0=True; labels=[r"$\Omega_\Lambda$"]
    elif case == "2D":
        flat=True; fix_H0=False; labels=[r"$\Omega_\Lambda$", r"$H_0$"]
    elif case == "3D":
        flat=False; fix_H0=False; labels=[r"$\Omega_\Lambda$", r"$\Omega_k$", r"$H_0$"]
    else:
        raise ValueError("case must be '1D','2D','3D'")

    # ------------------------
    # Run full dataset
    # ------------------------
    bf_chi2, _, _ = run_supernova_chi2(z, mu, sigma_mu, flat=flat, fix_H0=fix_H0, output=False)
    samples_mcmc, stats_mcmc = run_supernova_mcmc(z, mu, sigma_mu, flat=flat, fix_H0=fix_H0, output=False)

    # Convert chi2 best fits into array of "samples" by small perturbations to mimic covariance
    # (since χ² minimization doesn't provide full posterior, we use asym errors if available)
    # Here we just tile bf_chi2 for degeneracy metrics placeholder
    samples_chi2 = np.tile(bf_chi2, (100,1))  # simple placeholder; for full use, sample along Δχ² contours

    # ------------------------
    # Compute degeneracy metrics
    # ------------------------
    metrics_chi2 = degeneracy_metrics(samples_chi2, labels=labels)
    metrics_mcmc = degeneracy_metrics(samples_mcmc, labels=labels)

    if output:
        print("Chi-squared degeneracy metrics:")
        print("Covariance matrix:\n", metrics_chi2['covariance'])
        print("Correlation matrix:\n", metrics_chi2['correlation'])
        print("Eigenvalues:", metrics_chi2['eigenvalues'])
        print("Condition number:", metrics_chi2['condition_number'])
        print("\nMCMC degeneracy metrics:")
        print("Covariance matrix:\n", metrics_mcmc['covariance'])
        print("Correlation matrix:\n", metrics_mcmc['correlation'])
        print("Eigenvalues:", metrics_mcmc['eigenvalues'])
        print("Condition number:", metrics_mcmc['condition_number'])

        # ------------------------
        # 2D plots (if at least 2 parameters)
        # ------------------------
        if len(labels) >= 2:
            for i in range(len(labels)-1):
                plot_2d_degeneracy(samples_mcmc, i, i+1, labels[i], labels[i+1], method_name="MCMC", color='blue')
                plot_2d_degeneracy(samples_chi2, i, i+1, labels[i], labels[i+1], method_name="Chi-squared", color='red')

    return metrics_chi2, metrics_mcmc

# doesn't work for 1D
metrics_chi2, metrics_mcmc = compare_degeneracy(z, mu, sigma_mu, case="2D")
metrics_chi2, metrics_mcmc = compare_degeneracy(z, mu, sigma_mu, case="3D")
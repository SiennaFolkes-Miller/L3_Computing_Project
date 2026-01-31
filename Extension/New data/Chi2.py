import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from models import distance_modulus  # your single distance modulus model
from new_data_splitting import z, mu, sigma_mu  # your data arrays

# -----------------------------
# Chi-squared function
# -----------------------------
def chi2(theta, z, mu_obs, mu_err, flat=False):
    if flat:
        Omega_L, H0 = theta
        Omega_k = 0.0
    else:
        Omega_L, Omega_k, H0 = theta
        # Clip to prevent unphysical values
        Omega_L = np.clip(Omega_L, 0.0, 1.5)
        Omega_k = np.clip(Omega_k, -1.0, 1.0)
        H0 = max(H0, 1e-5)

    # Avoid z=0 to prevent log10(0)
    z_safe = np.maximum(z, 1e-5)

    mu_model = distance_modulus(z_safe, Omega_L, Omega_k, H0)
    # replace inf/nan with large number
    mu_model = np.where(np.isfinite(mu_model), mu_model, 1e6)

    return np.sum(((mu_obs - mu_model) / mu_err) ** 2)

# -----------------------------
# Function to compute asymmetric errors
# -----------------------------
def chi2_errors(best_fit, chi2_min, z, mu_obs, mu_err, flat=False, step=1e-3):
    """
    For each parameter, vary it until chi2 = chi2_min + 1
    Other parameters fixed.
    Returns (lower_error, upper_error) for each parameter
    """
    errors = []
    for i in range(len(best_fit)):
        param = best_fit[i]

        # positive direction
        val = param
        while True:
            val += step
            theta = best_fit.copy()
            theta[i] = val
            chi_val = chi2(theta, z, mu_obs, mu_err, flat=flat)
            if chi_val >= chi2_min + 1:
                upper = val - param
                break

        # negative direction
        val = param
        while True:
            val -= step
            theta = best_fit.copy()
            theta[i] = val
            chi_val = chi2(theta, z, mu_obs, mu_err, flat=flat)
            if chi_val >= chi2_min + 1:
                lower = param - val
                break

        errors.append((lower, upper))

    return errors

# -----------------------------
# Function to perform chi2 minimisation
# -----------------------------
def run_supernova_chi2(
    z,
    mu,
    sigma_mu,
    Omega_L_init=0.7,
    Omega_k_init=0.0,
    H0_init=70.0,
    flat=False
):
    # -----------------------------
    # Initial guess
    # -----------------------------
    if flat:
        p0 = [Omega_L_init, H0_init]
    else:
        p0 = [Omega_L_init, Omega_k_init, H0_init]

    # -----------------------------
    # Minimise chi-squared
    # -----------------------------
    result = minimize(
        chi2,
        p0,
        args=(z, mu, sigma_mu, flat),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'disp': True}
    )

    best_fit = result.x
    chi2_min = result.fun
    print("\nBest-fit parameters:")
    print(f"Minimum chi2 = {chi2_min:.4f}")

    # -----------------------------
    # Compute asymmetric errors
    # -----------------------------
    errors = chi2_errors(best_fit, chi2_min, z, mu, sigma_mu, flat=flat, step=1e-3)
    if flat:
        for name, (lo, hi) in zip(['Omega_L', 'H0'], errors):
            print(f"{name} = {best_fit[0 if name=='Omega_L' else 1]:.4g} -{lo:.4g} +{hi:.4g}")
    else:
        for name, (lo, hi), val in zip(['Omega_L', 'Omega_k', 'H0'], errors, best_fit):
            print(f"{name} = {val:.4g} -{lo:.4g} +{hi:.4g}")

    # -----------------------------
    # Contour plots (for 2D parameter pairs)
    # -----------------------------
    if flat:
        # Only Omega_L vs H0
        Omega_L_vals = np.linspace(best_fit[0]-0.2, best_fit[0]+0.2, 100)
        H0_vals = np.linspace(best_fit[1]-5, best_fit[1]+5, 100)
        chi2_grid = np.zeros((len(Omega_L_vals), len(H0_vals)))
        for i, O in enumerate(Omega_L_vals):
            for j, H in enumerate(H0_vals):
                chi2_grid[i, j] = chi2([O, H], z, mu, sigma_mu, flat=flat)

        # Contours at chi2_min + 1, 4, 9
        levels = [chi2_min + d for d in [1, 4, 9]]
        plt.contour(Omega_L_vals, H0_vals, chi2_grid.T, levels=levels, colors=['blue','green','red'])
        plt.xlabel(r"$\Omega_\Lambda$")
        plt.ylabel(r"$H_0$")
        plt.title("Chi2 Contours (flat universe)")
        plt.show()
    else:
    # ------------------------
    # Omega_L vs Omega_k (already existing)
    # ------------------------
        Omega_L_vals = np.linspace(best_fit[0]-0.4, best_fit[0]+0.4, 100)
        Omega_k_vals = np.linspace(best_fit[1]-0.8, best_fit[1]+0.8, 100)
        chi2_grid = np.zeros((len(Omega_L_vals), len(Omega_k_vals)))
        for i, O in enumerate(Omega_L_vals):
            for j, K in enumerate(Omega_k_vals):
                chi2_grid[i, j] = chi2([O, K, best_fit[2]], z, mu, sigma_mu, flat=flat)

        levels = [chi2_min + d for d in [1, 4, 9]]
        plt.contour(Omega_L_vals, Omega_k_vals, chi2_grid.T, levels=levels, colors=['blue','green','red'])
        plt.xlabel(r"$\Omega_\Lambda$")
        plt.ylabel(r"$\Omega_k$")
        plt.title("Chi2 Contours (curved universe, H0 fixed at best-fit)")
        plt.show()

    # ------------------------
    # Omega_k vs H0
    # ------------------------
        Omega_k_vals = np.linspace(best_fit[1]-0.8, best_fit[1]+0.8, 100)
        H0_vals = np.linspace(best_fit[2]-10, best_fit[2]+10, 100)
        chi2_grid = np.zeros((len(Omega_k_vals), len(H0_vals)))
        for i, K in enumerate(Omega_k_vals):
            for j, H in enumerate(H0_vals):
                chi2_grid[i, j] = chi2([best_fit[0], K, H], z, mu, sigma_mu, flat=flat)

        plt.contour(Omega_k_vals, H0_vals, chi2_grid.T, levels=levels, colors=['blue','green','red'])
        plt.xlabel(r"$\Omega_k$")
        plt.ylabel(r"$H_0$")
        plt.title("Chi2 Contours (curved universe, Omega_L fixed at best-fit)")
        plt.show()

    # ------------------------
    # Omega_L vs H0
    # ------------------------
        Omega_L_vals = np.linspace(best_fit[0]-0.4, best_fit[0]+0.4, 100)
        H0_vals = np.linspace(best_fit[2]-10, best_fit[2]+10, 100)
        chi2_grid = np.zeros((len(Omega_L_vals), len(H0_vals)))
        for i, O in enumerate(Omega_L_vals):
            for j, H in enumerate(H0_vals):
                chi2_grid[i, j] = chi2([O, best_fit[1], H], z, mu, sigma_mu, flat=flat)

        plt.contour(Omega_L_vals, H0_vals, chi2_grid.T, levels=levels, colors=['blue','green','red'])
        plt.xlabel(r"$\Omega_\Lambda$")
        plt.ylabel(r"$H_0$")
        plt.title("Chi2 Contours (curved universe, Omega_k fixed at best-fit)")
        plt.show()

    return best_fit, chi2_min, errors

# -----------------------------
# Example calls
# -----------------------------
# Flat universe
best_flat, chi2_flat, errors_flat = run_supernova_chi2(z, mu, sigma_mu, flat=True)

# Curved universe
best_curved, chi2_curved, errors_curved = run_supernova_chi2(z, mu, sigma_mu, flat=False)
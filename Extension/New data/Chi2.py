import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from models import distance_modulus  # your single distance modulus model
from new_data_splitting import z, mu, sigma_mu  # your data arrays

# -----------------------------
# Chi-squared function
# -----------------------------
def chi2(theta, z, mu_obs, mu_err, flat=False, fix_H0=False, H0_fixed=70.0):
    idx = 0
    Omega_L = theta[idx]; idx += 1
    Omega_k = 0.0 if flat else theta[idx]; idx += 1 if not flat else 0
    H0 = H0_fixed if fix_H0 else theta[idx]

    Omega_L = np.clip(Omega_L, 0.0, 1.5)
    Omega_k = np.clip(Omega_k, -1.0, 1.0)
    H0 = max(H0, 1e-5)

    z_safe = np.maximum(z, 1e-5)
    mu_model = distance_modulus(z_safe, Omega_L, Omega_k, H0)
    mu_model = np.where(np.isfinite(mu_model), mu_model, 1e6)
    return np.sum(((mu_obs - mu_model) / mu_err) ** 2)


# -----------------------------
# Asymmetric errors (Δχ² = 1)
# -----------------------------
def chi2_errors(best_fit, chi2_min, z, mu, sigma_mu, flat=False, fix_H0=False, H0_fixed=70.0, step=1e-3):
    errors = []
    n_params = len(best_fit)
    for i in range(n_params):
        param = best_fit[i]

        # Positive direction
        val = param
        while True:
            val += step
            theta = best_fit.copy()
            theta[i] = val
            chi_val = chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)
            if chi_val >= chi2_min + 1:
                upper = val - param
                break

        # Negative direction
        val = param
        while True:
            val -= step
            theta = best_fit.copy()
            theta[i] = val
            chi_val = chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)
            if chi_val >= chi2_min + 1:
                lower = param - val
                break

        errors.append((lower, upper))
    return errors


# -----------------------------
# Δχ²-based adaptive plotting ranges
# -----------------------------
def get_chi2_range(param_index, best_fit, chi2_min, z, mu, sigma_mu,
                   flat=False, fix_H0=False, H0_fixed=70.0, step=1e-3, delta_chi2=9,
                   pad_fraction=0.5):
    """
    Find parameter range for plotting such that Δχ² <= delta_chi2,
    and extend the range slightly for better visualization.
    """
    param_val = best_fit[param_index]

    # Positive direction
    val = param_val
    while True:
        val += step
        theta = best_fit.copy()
        theta[param_index] = val
        if chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed) - chi2_min >= delta_chi2:
            upper = val
            break

    # Negative direction
    val = param_val
    while True:
        val -= step
        theta = best_fit.copy()
        theta[param_index] = val
        if chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed) - chi2_min >= delta_chi2:
            lower = val
            break

    # Add padding
    range_span = upper - lower
    lower -= range_span * pad_fraction
    upper += range_span * pad_fraction

    return lower, upper


# -----------------------------
# Corner-style χ² contour plotting (Δχ²-adaptive)
# -----------------------------
def plot_chi2_corner(best_fit, chi2_min, z, mu, sigma_mu,
                     flat=False, fix_H0=False, H0_fixed=70.0, fine=100):

    # Parameter names
    names = [r"$\Omega_\Lambda$"]
    if not flat:
        names.append(r"$\Omega_k$")
    if not fix_H0:
        names.append(r"$H_0$")

    n_params = len(best_fit)

    # Compute adaptive ranges using Δχ² = 9
    ranges = []
    for i in range(n_params):
        low, high = get_chi2_range(i, best_fit, chi2_min, z, mu, sigma_mu,
                                   flat, fix_H0, H0_fixed, step=1e-3, delta_chi2=9)
        ranges.append((low, high))

    # 1D case
    if n_params == 1:
        x = np.linspace(ranges[0][0], ranges[0][1], fine)
        y = np.array([chi2([xi], z, mu, sigma_mu, flat, fix_H0, H0_fixed) for xi in x])
        plt.figure(figsize=(6,4))
        plt.plot(x, y)
        plt.axhline(chi2_min + 1, color='red', linestyle='--')
        plt.xlabel(names[0])
        plt.ylabel(r"$\chi^2$")
        plt.tight_layout()
        plt.show()
        return

    # 2D or 3D grids
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params,3*n_params))
    levels = [chi2_min + d for d in [1,4,9]]

    for i in range(n_params):
        for j in range(n_params):
            if n_params == 2:
                ax = axes[i,j]
            else:
                ax = axes[i,j]
            if i < j:
                ax.axis('off')
            elif i == j:
                # 1D projection
                x = np.linspace(ranges[i][0], ranges[i][1], fine)
                y = np.array([chi2([xk if l==i else best_fit[l] for l in range(n_params)],
                                   z, mu, sigma_mu, flat, fix_H0, H0_fixed) for xk in x])
                ax.plot(x, y)
                ax.axhline(chi2_min + 1, color='red', linestyle='--')
                ax.set_xlabel(names[i])
                ax.set_ylabel(r"$\chi^2$")
            else:
                # 2D contour
                x = np.linspace(ranges[j][0], ranges[j][1], fine)
                y = np.linspace(ranges[i][0], ranges[i][1], fine)
                X, Y = np.meshgrid(x, y)
                chi_grid = np.zeros_like(X)
                for ix in range(fine):
                    for iy in range(fine):
                        theta = best_fit.copy()
                        theta[j] = X[iy, ix]
                        theta[i] = Y[iy, ix]
                        chi_grid[iy, ix] = chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)
                ax.contour(x, y, chi_grid, levels=levels, colors=['blue','green','red'])
                ax.set_xlabel(names[j])
                ax.set_ylabel(names[i])

    plt.tight_layout()
    plt.show()

def run_supernova_chi2(z, mu, sigma_mu, Omega_L_init=0.7, Omega_k_init=0.0, H0_init=70.0,
                        flat=False, fix_H0=False, H0_fixed=70.0):
    # Initial guess
    p0 = [Omega_L_init]
    if not flat:
        p0.append(Omega_k_init)
    if not fix_H0:
        p0.append(H0_init)

    # Minimise
    result = minimize(chi2, p0, args=(z, mu, sigma_mu, flat, fix_H0, H0_fixed),
                      method='Nelder-Mead', options={'maxiter':10000, 'disp': True})
    best_fit = result.x
    chi2_min = result.fun

    # Asymmetric errors
    errors = chi2_errors(best_fit, chi2_min, z, mu, sigma_mu, flat, fix_H0, H0_fixed)

    # Print results
    print("\nBest-fit parameters:")
    for name, val, (lo, hi) in zip(['Omega_L', 'Omega_k', 'H0'][:len(best_fit)], best_fit, errors):
        print(f"{name} = {val:.4g} -{lo:.4g} +{hi:.4g}")
    print(f"Minimum chi2 = {chi2_min:.4f}")

    # Plot corner-style contours
    plot_chi2_corner(best_fit, chi2_min, z, mu, sigma_mu, flat, fix_H0, H0_fixed)

    return best_fit, chi2_min, errors
# -----------------------------
# Example calls
# -----------------------------
# 1D: flat + H0 fixed
#best_flat_1D, chi2_flat_1D, errors_flat_1D = run_supernova_chi2(z, mu, sigma_mu, flat=True, fix_H0=True)

# 2D: flat
best_flat_2D, chi2_flat_2D, errors_flat_2D = run_supernova_chi2(z, mu, sigma_mu, flat=True, fix_H0=False)

# 2D: curved + H0 fixed
#best_curve_2D, chi2_curve_2D, errors_curve_2D = run_supernova_chi2(z, mu, sigma_mu, flat=False, fix_H0=True)

# 3D: curved
#best_curve_3D, chi2_curve_3D, errors_curve_3D = run_supernova_chi2(z, mu, sigma_mu, flat=False, fix_H0=False)

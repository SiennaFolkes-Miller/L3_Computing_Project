import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

# -----------------------------
# Import your own modules/data
# -----------------------------
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
# Profile-likelihood asymmetric errors
# -----------------------------
def chi2_profile(param_index, best_fit, z, mu, sigma_mu,
                 flat=False, fix_H0=False, H0_fixed=70.0,
                 delta_chi2=1.0, step=1e-3, max_iter=5000):
    n_params = len(best_fit)
    theta0 = best_fit.copy()
    chi2_min = chi2(theta0, z, mu, sigma_mu, flat, fix_H0, H0_fixed)

    free_indices = [i for i in range(n_params) if i != param_index]

    # Direct scan if no other free parameters
    if len(free_indices) == 0:
        # Negative direction
        val = theta0[param_index]
        while True:
            val -= step
            chi_val = chi2([val], z, mu, sigma_mu, flat, fix_H0, H0_fixed)
            if chi_val - chi2_min >= delta_chi2:
                lower = theta0[param_index] - val
                break

        # Positive direction
        val = theta0[param_index]
        while True:
            val += step
            chi_val = chi2([val], z, mu, sigma_mu, flat, fix_H0, H0_fixed)
            if chi_val - chi2_min >= delta_chi2:
                upper = val - theta0[param_index]
                break

        return lower, upper

    # Scan negative direction with profiling
    lower = None
    val = theta0[param_index]
    for _ in range(int(1e5)):
        val -= step

        def profile_fn(p_free):
            theta = theta0.copy()
            for idx, pf in zip(free_indices, p_free):
                theta[idx] = pf
            theta[param_index] = val
            return chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)

        p0_free = [theta0[idx] for idx in free_indices]
        res = minimize(profile_fn, p0_free, method='Powell',
                       options={'maxiter': max_iter, 'disp': False, 'xtol':1e-6, 'ftol':1e-6})
        chi_val = res.fun
        if chi_val - chi2_min >= delta_chi2:
            lower = theta0[param_index] - val
            break

    # Scan positive direction with profiling
    upper = None
    val = theta0[param_index]
    for _ in range(int(1e5)):
        val += step

        def profile_fn(p_free):
            theta = theta0.copy()
            for idx, pf in zip(free_indices, p_free):
                theta[idx] = pf
            theta[param_index] = val
            return chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)

        p0_free = [theta0[idx] for idx in free_indices]
        res = minimize(profile_fn, p0_free, method='Powell',
                       options={'maxiter': max_iter, 'disp': False, 'xtol':1e-6, 'ftol':1e-6})
        chi_val = res.fun
        if chi_val - chi2_min >= delta_chi2:
            upper = val - theta0[param_index]
            break

    return lower, upper

def chi2_errors_profile(best_fit, z, mu, sigma_mu, flat=False, fix_H0=False, H0_fixed=70.0):
    errors = []
    for i in range(len(best_fit)):
        lo, hi = chi2_profile(i, best_fit, z, mu, sigma_mu, flat, fix_H0, H0_fixed)
        errors.append((lo, hi))
    return errors

# -----------------------------
# Δχ²-based adaptive plotting ranges
# -----------------------------
def get_chi2_range(param_index, best_fit, chi2_min, z, mu, sigma_mu,
                   flat=False, fix_H0=False, H0_fixed=70.0,
                   step=1e-3, delta_chi2=9.0, pad_fraction=0.5, max_scan=10):
    param_val = best_fit[param_index]

    # Positive direction
    val = param_val
    for _ in range(int(max_scan/step)):
        val += step
        theta = best_fit.copy()
        theta[param_index] = val
        if chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed) - chi2_min >= delta_chi2:
            upper = val
            break
    else:
        upper = val + 2*step

    # Negative direction
    val = param_val
    for _ in range(int(max_scan/step)):
        val -= step
        theta = best_fit.copy()
        theta[param_index] = val
        if chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed) - chi2_min >= delta_chi2:
            lower = val
            break
    else:
        lower = val - 2*step

    # Add padding
    range_span = upper - lower
    lower -= range_span * pad_fraction
    upper += range_span * pad_fraction

    return lower, upper

# -----------------------------
# Corner-style χ² plotting with profile-likelihood for 1D
# -----------------------------
def plot_chi2_corner(best_fit, chi2_min, z, mu, sigma_mu,
                     flat=False, fix_H0=False, H0_fixed=70.0, fine=30):
    """
    Corner-style χ² plots with Δχ²-adaptive ranges.

    Diagonal 1D plots: profile-likelihood χ² (other parameters free).
    Off-diagonal 2D plots: standard χ² contours.

    Parameters
    ----------
    best_fit : array-like
        Best-fit parameter values.
    chi2_min : float
        Minimum χ² value.
    z, mu, sigma_mu : arrays
        Supernova data.
    flat : bool
        Fix Omega_k = 0.
    fix_H0 : bool
        Fix H0.
    H0_fixed : float
        H0 value if fixed.
    fine : int
        Number of points along each axis.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from scipy.interpolate import UnivariateSpline

    # -----------------------------
    # Parameter names
    # -----------------------------
    names = [r"$\Omega_\Lambda$"]
    if not flat:
        names.append(r"$\Omega_k$")
    if not fix_H0:
        names.append(r"$H_0$")

    n_params = len(best_fit)

    # -----------------------------
    # Δχ² levels for confidence regions
    # -----------------------------
    if n_params == 1:
        levels = [chi2_min + 1, chi2_min + 4, chi2_min + 9]
    elif n_params == 2:
        levels = [chi2_min + 2.30, chi2_min + 6.17, chi2_min + 11.8]
    else:
        levels = [chi2_min + 3.53, chi2_min + 8.02, chi2_min + 14.2]

    # -----------------------------
    # Compute adaptive ranges (Δχ²-based)
    # -----------------------------
    ranges = []
    for i in range(n_params):
        low, high = get_chi2_range(i, best_fit, chi2_min, z, mu, sigma_mu,
                                   flat, fix_H0, H0_fixed,
                                   delta_chi2=levels[-1]-chi2_min)
        ranges.append((low, high))

    axis_lims = ranges.copy()  # store for consistent axis limits

    # -----------------------------
    # Handle 1D case separately
    # -----------------------------
    if n_params == 1:
        x = np.linspace(ranges[0][0], ranges[0][1], fine)
        y = np.zeros_like(x)
        for ix, xi in enumerate(x):
            theta = best_fit.copy()
            theta[0] = xi
            y[ix] = chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)
        y_smooth = UnivariateSpline(x, y, s=0.0)
        plt.figure(figsize=(6,4))
        plt.plot(x, y_smooth(x))
        delta_chi2_1sigma = levels[0] - chi2_min
        plt.axhline(chi2_min + delta_chi2_1sigma, color='red', linestyle='--',
                    label=fr"$\Delta \chi^2 = {delta_chi2_1sigma:.2f}$")
        plt.xlabel(names[0])
        plt.ylabel(r"$\chi^2$")
        plt.title("χ² Profile")
        plt.tight_layout()
        plt.show()
        return

    # -----------------------------
    # 2D/3D grids
    # -----------------------------
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i,j] if n_params > 1 else axes

            if i < j:
                ax.axis('off')
            elif i == j:
                # -----------------------------
                # 1D profile-likelihood χ²
                # -----------------------------
                x_vals = np.linspace(ranges[i][0], ranges[i][1], fine)
                y_vals = np.zeros_like(x_vals)

                for ix, xval in enumerate(x_vals):
                    free_indices = [k for k in range(n_params) if k != i]
                    if len(free_indices) == 0:
                        theta = best_fit.copy()
                        theta[i] = xval
                        y_vals[ix] = chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)
                    else:
                        def profile_fn(p_free):
                            theta = best_fit.copy()
                            for idx, pf in zip(free_indices, p_free):
                                theta[idx] = pf
                            theta[i] = xval
                            return chi2(theta, z, mu, sigma_mu, flat, fix_H0, H0_fixed)
                        p0_free = [best_fit[idx] for idx in free_indices]
                        res = minimize(profile_fn, p0_free, method='Powell',
                                       options={'maxiter':5000, 'disp':False,'xtol':1e-6,'ftol':1e-6})
                        y_vals[ix] = res.fun

                y_smooth = UnivariateSpline(x_vals, y_vals, s=0.0)
                ax.plot(x_vals, y_smooth(x_vals))
                delta_chi2_1sigma = levels[0] - chi2_min
                ax.axhline(chi2_min + delta_chi2_1sigma, color='red', linestyle='--',
                           label=fr"$\Delta \chi^2 = {delta_chi2_1sigma:.2f}$")
                ax.set_xlabel(names[i])
                ax.set_ylabel(r"$\chi^2$")
                ax.set_xlim(axis_lims[i])
            else:
                # -----------------------------
                # 2D χ² contour
                # -----------------------------
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

                ax.contour(X, Y, chi_grid, levels=levels, colors=['blue','green','red'])
                ax.set_xlabel(names[j])
                ax.set_ylabel(names[i])
                ax.set_xlim(axis_lims[j])
                ax.set_ylim(axis_lims[i])

    plt.tight_layout()
    plt.show()

# -----------------------------
# Main χ² runner
# -----------------------------
def run_supernova_chi2(z, mu, sigma_mu,
                       Omega_L_init=0.7, Omega_k_init=0.0, H0_init=70.0,
                       flat_universe=False, fix_H0=False, H0_fixed=70.0,
                       output=True):
    p0 = [Omega_L_init]
    if not flat_universe:
        p0.append(Omega_k_init)
    if not fix_H0:
        p0.append(H0_init)

    result = minimize(
        chi2, p0,
        args=(z, mu, sigma_mu, flat_universe, fix_H0, H0_fixed),
        method='Nelder-Mead',
        options={'maxiter':10000, 'disp': True}
    )
    best_fit = result.x
    chi2_min = result.fun

    errors = chi2_errors_profile(best_fit, z, mu, sigma_mu, flat=flat_universe, fix_H0=fix_H0, H0_fixed=H0_fixed)

    dof = len(z) - len(best_fit)
    chi2_red = chi2_min / dof

    if output:
        names = ['Omega_L', 'H0', 'Omega_k'][:len(best_fit)]
        print("\nBest-fit parameters (profile likelihood errors):")
        for name, val, (lo, hi) in zip(names, best_fit, errors):
            print(f"{name} = {val:.4g} -{lo:.4g} +{hi:.4g}")
        print(f"Minimum chi2 / dof = {chi2_red:.4f}")

        plot_chi2_corner(best_fit, chi2_min, z, mu, sigma_mu, flat_universe, fix_H0, H0_fixed)

    return best_fit, chi2_red, errors

# -----------------------------
# Example calls
# -----------------------------
# 1D: flat + H0 fixed
best_flat_1D, chi2_flat_1D, errors_flat_1D = run_supernova_chi2(
    z, mu, sigma_mu, flat_universe=True, fix_H0=True, output=True
)

# 2D: flat
best_flat_2D, chi2_flat_2D, errors_flat_2D = run_supernova_chi2(
    z, mu, sigma_mu, flat_universe=True, fix_H0=False, output=True
)

# 3D: curved
best_curve_3D, chi2_curve_3D, errors_curve_3D = run_supernova_chi2(z, mu, sigma_mu, flat_universe=False, fix_H0=False, output=True)

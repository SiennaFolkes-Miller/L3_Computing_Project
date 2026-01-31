import numpy as np
from scipy.optimize import minimize
from model_flat import m_model
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from old_data import low_redshift_values, high_redshift_values
data_all = np.vstack([low_redshift_values, high_redshift_values])  
z = data_all[:, 0]           
m = data_all[:, 1]           
sigma_m = data_all[:, 2] 

#z, m, sigma_m = load_scp_data("SCP_data.tex")


def m_model_curvefit(z, Omega_L, L):
    return m_model(z, Omega_L, L)

# --- Chi-squared function ---
def chi2(Omega_L, L, z, m, sigma_m):
    m_model_vals = m_model_curvefit(z, Omega_L, L)
    return np.sum(((m - m_model_vals) / sigma_m)**2)

# --- Modified function ---
def fit_supernova_curvefit(Omega_L_init=0.7, L_init=3e32):
    # Initial guess
    p0 = [Omega_L_init, L_init]
    
    # Fit using curve_fit
    popt, pcov = curve_fit(
        m_model_curvefit, z, m, sigma=sigma_m,
        p0=p0, absolute_sigma=True
    )
    
    Omega_L_best, L_best = popt
    sigma_Omega_L, sigma_L = np.sqrt(np.diag(pcov))
    
    print(f"Best-fit Omega_L = {Omega_L_best:.4f} ± {sigma_Omega_L:.4f}")
    print(f"Best-fit L       = {L_best:.3e} ± {sigma_L:.3e}")
    
    # Optional: plot data vs model
    z_plot = np.linspace(0, np.max(z), 200)
    m_fit = m_model_curvefit(z_plot, Omega_L_best, L_best)
    
    plt.errorbar(z, m, yerr=sigma_m, fmt='o', label='Data')
    plt.plot(z_plot, m_fit, label='Best-fit model')
    plt.xlabel("Redshift z")
    plt.ylabel("Apparent magnitude m")
    plt.legend()
    plt.show()
    
    # --- Chi-squared contour plot ---
    Omega_L_vals = np.linspace(Omega_L_best - 0.2, Omega_L_best + 0.2, 100)
    L_vals = np.linspace(L_best * 0.5, L_best * 1.5, 500)
    chi2_grid = np.zeros((len(Omega_L_vals), len(L_vals)))
    
    for i, Omega_L_i in enumerate(Omega_L_vals):
        for j, L_j in enumerate(L_vals):
            chi2_grid[i, j] = chi2(Omega_L_i, L_j, z, m, sigma_m)
    
    chi2_min = np.min(chi2_grid)
    
    # Delta chi-squared levels: 1, 4, 9
    delta_chi2_levels = [1, 4, 9]
    levels = [chi2_min + dchi for dchi in delta_chi2_levels]
    
    contour_colors = ['blue', 'green', 'lime']
    
    cs = plt.contour(Omega_L_vals, L_vals, chi2_grid.T, levels=levels, colors=contour_colors)
    
    # Label contours
    fmt = {level: f'Δχ²={int(level - chi2_min)}' for level in levels}
    plt.clabel(cs, inline=True, fmt=fmt, fontsize=10)

    plt.plot(Omega_L_best, L_best, 'kx', color = 'red', label='Best fit')
    plt.xlabel(r'$\Omega_\Lambda$')
    plt.ylabel(r'$L$')
    plt.title(r'$\chi^2$ Contours')
    plt.legend()
    plt.show()
    
    return Omega_L_best, L_best, sigma_Omega_L, sigma_L

# Example call
Omega_L_best, L_best, sigma_Omega_L, sigma_L = fit_supernova_curvefit()
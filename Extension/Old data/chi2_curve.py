import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from old_data import low_redshift_values, high_redshift_values
from model_curve import m_model_curved  # This should be the curved-universe version of m_model

# Combine low-z and high-z data
data_all = np.vstack([low_redshift_values, high_redshift_values])
z = data_all[:, 0]
m = data_all[:, 1]
sigma_m = data_all[:, 2]

# --- Wrapper for curve_fit ---
def m_model_curvefit_safe(z, Omega_L, Omega_k, L):
    Omega_L = np.clip(Omega_L, 0.0, 1.5)
    Omega_k = np.clip(Omega_k, -1.0, 1.0)
    L = max(L, 1e30)  # prevent zero
    m = m_model_curved(z, Omega_L, Omega_k, L)
    # replace any remaining inf/nan
    m = np.where(np.isfinite(m), m, 1e6)
    return m

# --- Function to fit supernova data ---
def fit_supernova_curvefit_curved(Omega_L_init=0.7, Omega_k_init=0.0, L_init=3e32):
    # Initial guess
    p0 = [Omega_L_init, Omega_k_init, L_init]
    
    # Fit using curve_fit
    p0 = [0.7, 0.0, 3e32]  # Omega_L, Omega_k, L
    bounds = ([0.0, -1.0, 0.0], [1.5, 1.0, np.inf])

    popt, pcov = curve_fit(
        m_model_curvefit_safe,
        z,
        m,
        sigma=sigma_m,
        p0=p0,
        absolute_sigma=True,
        bounds=bounds
    )
    
    # Extract best-fit parameters
    Omega_L_best, Omega_k_best, L_best = popt
    sigma_Omega_L, sigma_Omega_k, sigma_L = np.sqrt(np.diag(pcov))
    
    print(f"Best-fit Omega_L = {Omega_L_best:.4f} ± {sigma_Omega_L:.4f}")
    print(f"Best-fit Omega_k = {Omega_k_best:.4e} ± {sigma_Omega_k:.4e}")
    print(f"Best-fit L       = {L_best:.3e} ± {sigma_L:.3e}")
    
    # Optional: plot data vs model
    z_plot = np.linspace(0, np.max(z), 200)
    m_fit = m_model_curvefit_safe(z_plot, Omega_L_best, Omega_k_best, L_best)
    
    plt.errorbar(z, m, yerr=sigma_m, fmt='o', label='Data')
    plt.plot(z_plot, m_fit, label='Best-fit curved model')
    plt.xlabel("Redshift z")
    plt.ylabel("Apparent magnitude m")
    plt.legend()
    plt.show()
    
    return Omega_L_best, Omega_k_best, L_best, sigma_Omega_L, sigma_Omega_k, sigma_L

# Run the fit
Omega_L_best, Omega_k_best, L_best, sigma_Omega_L, sigma_Omega_k, sigma_L = fit_supernova_curvefit_curved()
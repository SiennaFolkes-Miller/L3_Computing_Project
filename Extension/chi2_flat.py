import numpy as np
from scipy.optimize import minimize
from new_data_splitting import load_scp_data
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

# --- Function to fit supernova data ---
def fit_supernova_curvefit(Omega_L_init=0.7, L_init=3e32):
    
    # Initial guess
    p0 = [Omega_L_init, L_init]
    
    # Fit using curve_fit
    popt, pcov = curve_fit(
        m_model_curvefit, z, m, sigma=sigma_m,
        p0=p0, absolute_sigma=True
    )
    
    # Best-fit parameters
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
    
    return Omega_L_best, L_best, sigma_Omega_L, sigma_L

Omega_L_best, L_best, sigma_Omega_L, sigma_L = fit_supernova_curvefit()
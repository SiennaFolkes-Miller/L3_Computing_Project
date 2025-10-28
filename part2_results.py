import numpy as np
import astropy.units as u
from astropy.constants import c
from data_splitting import high_redshift_values
from part2_defs import H_of_z
from part2_defs import luminosity_distance
from part2_defs import m_model
from part2_defs import chi_squared

Omega_L_values = np.linspace(0.1, 1, 500) 

#runs chi squared minimisation over given L values
chi2_values = np.array([chi_squared(O, high_redshift_values)[0] for O in Omega_L_values])

#finds lowest chi2 and returns matching L and chired
best_index = np.argmin(chi2_values)
best_Omega_L = Omega_L_values[best_index]
chi2_min = chi2_values[best_index]
chi2_red = chi_squared(best_Omega_L, high_redshift_values)[1]

#finds chi2min+1 boundaries and their corresponding Ls to find Lerr
chi2_threshold = chi2_min + 1
Omega_L_low = np.interp(chi2_threshold, chi2_values[:best_index][::-1], Omega_L_values[:best_index][::-1])
Omega_L_high = np.interp(chi2_threshold, chi2_values[best_index:], Omega_L_values[best_index:])
sigma_OL_minus = best_Omega_L - Omega_L_low
sigma_OL_plus = Omega_L_high - best_Omega_L
average_error = (sigma_OL_minus + sigma_OL_plus) / 2

#editing s.f. and d.p. for L and its error
error_rounded = float(f"{average_error:.1g}") 
decimal_places = -int(np.floor(np.log10(error_rounded)))
Omega_L_rounded = round(best_Omega_L, decimal_places)

print(f"Best-fit Omega L = ({Omega_L_rounded:.{decimal_places}f} ± {error_rounded:.{decimal_places}f})")
print(f"Minimum chi-squared = {chi2_min}")
print(f"Reduced chi-squared = {chi2_red}")
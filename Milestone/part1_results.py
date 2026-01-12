import numpy as np
import astropy.units as u
from astropy.constants import c
from data_splitting import low_redshift_values
from part1_defs import m_model
from part1_defs import chi_squared
from part1_defs import H0

L_values = np.logspace(32, 33, 500) * u.W / u.AA

#runs chi squared minimisation over given L values
chi2_values = np.array([chi_squared(L, low_redshift_values)[0] for L in L_values])

#finds lowest chi2 and returns matching L and chired
best_index = np.argmin(chi2_values)
best_L = L_values[best_index]
chi2_min = chi2_values[best_index]
chi2_red = chi_squared(best_L, low_redshift_values)[1]

#finds chi2min+1 boundaries and their corresponding Ls to find Lerr
chi2_threshold = chi2_min + 1
L_low = np.interp(chi2_threshold, chi2_values[:best_index][::-1], L_values[:best_index][::-1].value) * L_values.unit
L_high = np.interp(chi2_threshold, chi2_values[best_index:], L_values[best_index:].value) * L_values.unit
sigma_L_minus = best_L - L_low
sigma_L_plus = L_high - best_L
average_error = (sigma_L_minus + sigma_L_plus) / 2

print(best_L, average_error)
#editing s.f. and d.p. for L and its error
exponent = int(np.floor(np.log10(best_L.value)))
scaled_L = best_L.value / 10**exponent
scaled_error = average_error.value / 10**exponent
scaled_error_rounded = float(f"{scaled_error:.2g}")
decimal_places = -int(np.floor(np.log10(scaled_error_rounded))-1)
scaled_L_rounded = round(scaled_L, decimal_places)

print(f"Best-fit L = ({scaled_L_rounded:.{decimal_places}f} ± {scaled_error_rounded:.{decimal_places}f})e{exponent} {best_L.unit}")
print(f"Minimum chi-squared = {chi2_min}")
print(f"Reduced chi-squared = {chi2_red}")

#L in terms of solar luminosity (take as order of magnitude estimate)
L_sun = 3.828*10**26*u.W/(1000*u.AA)   #have guessed 1000 as typical angstrom range for B band
L_over_Ls = best_L/L_sun
LoLs_error = average_error/L_sun
exponent = int(np.floor(np.log10(L_over_Ls.value)))
scaled_L = L_over_Ls.value / 10**exponent
scaled_error = LoLs_error.value / 10**exponent
scaled_error_rounded = float(f"{scaled_error:.2g}")
decimal_places = -int(np.floor(np.log10(scaled_error_rounded)))
scaled_L_rounded = round(scaled_L, decimal_places)
print(f"L/Lsun = ({scaled_L_rounded:.{decimal_places}f} ± {scaled_error_rounded:.{decimal_places}f})e{exponent} {L_over_Ls.unit}")

#editing s.f. and d.p. for LH2 and its error
LH2 = (best_L*H0**2).to(u.W / (u.m * u.s**2))
LH2_error = (average_error*H0**2).to(u.W / (u.m * u.s**2))
exponent = int(np.floor(np.log10(LH2.value)))
scaled_L = LH2.value / 10**exponent
scaled_error = LH2_error.value / 10**exponent
scaled_error_rounded = float(f"{scaled_error:.2g}")
decimal_places = -int(np.floor(np.log10(scaled_error_rounded)))
scaled_L_rounded = round(scaled_L, decimal_places)
print(f"Lₚₑₐₖ × H₀² = ({scaled_L_rounded:.{decimal_places}f} ± {scaled_error_rounded:.{decimal_places}f})e{exponent} {LH2.unit}")

import numpy as np
import astropy.units as u
from astropy.constants import c
import matplotlib.pyplot as plt
from data_splitting import low_redshift_values
print(len(low_redshift_values))
H0 = 75 * (u.km / u.s / u.Mpc)
f0 = 6.61e-9 * u.erg / (u.cm**2 * u.s * u.AA)

#luminosity is currently units erg/sA  
def m_model(z, L):
    #takes redshift and peak luminosity and returns model effective magnitude
    dL = (c * z / H0).to(u.cm)
    f = L / (4 * np.pi * dL**2) #note have ignored extra factor of (1+z)
    m = -2.5 * np.log10(f / f0)  #note have ignored extra factor of (1+z)
    return m.value  

def chi_squared(L, data):
    #takes model L value in erg/sA and observed m, err and z data and runs chi-squared minimisation and returns min chi2
    z = data[:,0]
    m_obs = data[:,1]
    err = data[:,2]
    
    m_mod = m_model(z, L)
    chi2 = np.sum(((m_obs - m_mod) / err)**2)
    return chi2

L_values = np.logspace(5, 55, 500) * u.erg / u.s / u.AA  
#runs chi squared minimisation over given L values
chi2_values = np.array([chi_squared(L, low_redshift_values) for L in L_values])

#finds lowest chi2 and returns matching L
best_index = np.argmin(chi2_values)
best_L = L_values[best_index]

print(f"Best-fit L = {best_L}")
print(f"Best-fit L in W/A = {best_L.to(u.W/u.AA)}")
print(f"Minimum chi-squared = {chi2_values[best_index]}")

z_obs = low_redshift_values[:,0]
m_obs = low_redshift_values[:,1]
err = low_redshift_values[:,2]


fig, ax = plt.subplots(2,1, figsize=(8,8), sharex=True, gridspec_kw={'height_ratios':[3,1]})
#plots observed data
ax[0].errorbar(z_obs, m_obs, yerr=err, fmt='o', label='Observed', capsize=3)

#plots model with best L and lowest chi2
z_model = np.linspace(min(z_obs)*0.9, max(z_obs)*1.1, 200)
m_model_vals = m_model(z_model, best_L)
ax[0].plot(z_model, m_model_vals, color='red', label='Best-fit model', lw=2)
#plt.legend()

#plots residuals
residuals = np.array((m_obs - m_model(z_obs, best_L))/err)
ax[1].scatter(z_obs, residuals, color='magenta')
ax[1].axhline(y=0, color='black')
ax[1].axhline(y=1, color='grey', linestyle='--')
ax[1].axhline(y=-1, color='grey', linestyle='--')

plt.xlabel('Redshift (z)')
ax[0].set_ylabel('Effective magnitude (m)')
ax[1].set_ylabel('Residuals')
fig.subplots_adjust(hspace=0)
plt.show()
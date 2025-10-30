import numpy as np
import astropy.units as u
from astropy.constants import c
from data_splitting import low_redshift_values

H0 = 75 * (u.km / u.s / u.Mpc)
f0 = 6.61e-12 * u.W / (u.m**2 * u.AA)

#luminosity is in W/Å
def m_model(z, L):
    #takes redshift and peak luminosity and returns model effective magnitude
    dL = (c * z / H0).to(u.m)
    f = L / (4 * np.pi * dL**2) #note have ignored extra factor of (1+z)
    m = -2.5 * np.log10(f / f0)  #note have ignored extra factor of (1+z)
    return m.value  

def chi_squared(L, data):
    #takes model L value and observed m, err and z data and runs chi-squared minimisation and returns min chi2 and chired
    z = data[:,0]
    m_obs = data[:,1]
    err = data[:,2]
    
    m_mod = m_model(z, L)
    chi2 = np.sum(((m_obs - m_mod) / err)**2)
    N = len(m_obs)  # number of data points
    chi2_red = chi2/(N-1)
    
    return chi2, chi2_red
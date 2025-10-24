import numpy as np
import astropy.units as u
from astropy.constants import c
from scipy.integrate import cumulative_trapezoid as cumtrapz
from data_splitting import high_redshift_values
from part1_results import best_L

H0 = 75 * (u.km / u.s / u.Mpc)
f0 = 6.61e-12 * u.W / (u.m**2 * u.AA)
L = best_L

#Hubble parameter, takes z and dark energy density parameter
def H_of_z(z, Omega_L):
    return H0*np.sqrt((1.0 - Omega_L)*(1 + z)**3 + Omega_L)

#Luminosity distance, needs array of redshifts
def luminosity_distance(z, Omega_L):
    z_grid = np.linspace(0, np.max(z), 2000)

    Hz = H_of_z(z_grid, Omega_L)
    Hz_SI = Hz.to(1/u.s)
    inv_H = (1 / Hz_SI).value #in seconds
    I_grid = cumtrapz(inv_H, z_grid, initial=0.0)
    I_interp = np.interp(z, z_grid, I_grid)
    D_L = c * I_interp * u.m
    return D_L

#L is now W/Å
def m_model(z, Omega_L):
    D_L = luminosity_distance(z, Omega_L)
    f = (L / (4 * np.pi * D_L**2)) # W / (m² Å)
    m = -2.5 * np.log10((f / f0).value)
    return m

#same as before
def chi_squared(Omega_L, data):
    z = data[:, 0]
    m_obs = data[:, 1]
    err = data[:, 2]

    m_mod = m_model(z, Omega_L)
    chi2 = np.sum(((m_obs - m_mod) / err)**2)
    N = len(m_obs)
    chi2_red = chi2/N
    return chi2, chi2_red


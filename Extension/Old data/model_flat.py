import numpy as np
import astropy.units as u
from astropy.constants import c
from scipy.integrate import cumulative_trapezoid as cumtrapz

H0 = 75 * (u.km / u.s / u.Mpc) 
#H0 = 67.36 * (u.km / u.s / u.Mpc) #from CMB prior paper, plus minus 0.54
f0 = 6.61e-12 * u.W / (u.m**2 * u.AA)

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
    D_L = c * I_interp * (1+z)
    return D_L

#L is now W/Å
def m_model(z, Omega_L, L):
    D_L = luminosity_distance(z, Omega_L)
    f = L / (4 * np.pi * D_L**2)  # W / (m² Å)
    m = -2.5 * np.log10((f / f0).value)
    return m

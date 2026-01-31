import numpy as np
import astropy.units as u
from astropy.constants import c
from scipy.integrate import cumulative_trapezoid as cumtrapz

#simply set omega_k = 0 for flat universe tests
def H_of_z(z, Omega_L, Omega_k, H0):
    # attach units once
    H0_q = H0 * u.km / u.s / u.Mpc

    Omega_m = 1.0 - Omega_L - Omega_k

    return H0_q * np.sqrt(
        Omega_m * (1 + z)**3 +
        Omega_k * (1 + z)**2 +
        Omega_L
    )

def comoving_distance(z, Omega_L, Omega_k, H0):
    z = np.asarray(z)
    z_grid = np.linspace(0, np.max(z), 2000)

    Hz = H_of_z(z_grid, Omega_L, Omega_k, H0)      # H0 as float
    inv_H = (1.0 / Hz.to(1/u.s)).value             # seconds

    integral = cumtrapz(inv_H, z_grid, initial=0.0)
    integral_interp = np.interp(z, z_grid, integral)

    # convert to Mpc
    return (c * integral_interp * u.s).to(u.Mpc)

def transverse_comoving_distance(z, Omega_L, Omega_k, H0):
    D_C = comoving_distance(z, Omega_L, Omega_k, H0)

    if np.isclose(Omega_k, 0.0):
        return D_C

    sqrt_ok = np.sqrt(np.abs(Omega_k))
    H0_q = H0 * u.km / u.s / u.Mpc
    H0_si = H0_q.to(1/u.s)
    factor = (c / H0_si).to(u.Mpc)   # length units

    # extract numeric values for trig functions
    D_C_val = D_C.value
    factor_val = factor.value

    if Omega_k > 0.0:
        return (factor / sqrt_ok) * np.sinh(sqrt_ok * D_C_val / factor_val)
    else:
        return (factor / sqrt_ok) * np.sin(sqrt_ok * D_C_val / factor_val)

def luminosity_distance(z, Omega_L, Omega_k, H0):
    D_M = transverse_comoving_distance(z, Omega_L, Omega_k, H0)
    return (1 + z) * D_M

def distance_modulus(z, Omega_L, Omega_k, H0):
    D_L = luminosity_distance(z, Omega_L, Omega_k, H0)
    D_L_pc = D_L.to(u.pc).value
    return 5 * np.log10(D_L_pc) - 5
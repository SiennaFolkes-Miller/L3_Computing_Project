import numpy as np
import astropy.units as u
from astropy.constants import c
from scipy.integrate import cumulative_trapezoid as cumtrapz

H0 = 75 * (u.km / u.s / u.Mpc) 
#H0 = 67.36 * (u.km / u.s / u.Mpc) #from CMB prior paper, plus minus 0.54
f0 = 6.61e-12 * u.W / (u.m**2 * u.AA)  # reference flux for magnitudes

# ----------------------------------------
# 1️⃣ Hubble parameter H(z) with curvature
# ----------------------------------------
def H_of_z_curved(z, Omega_L, Omega_k):
    Omega_m = 1.0 - Omega_L - Omega_k
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_L)

# ----------------------------------------
# 2️⃣ Comoving distance helper for curvature
# ----------------------------------------
def S_k(x, Omega_k):
    # Flat case
    if np.isclose(Omega_k, 0):
        return x
    # Open universe (Omega_k > 0)
    elif Omega_k > 0:
        sqrt_ok = np.sqrt(Omega_k)
        arg = sqrt_ok * x
        # Use asymptotic formula for large arg
        # sinh(arg) ~ 0.5 * exp(arg)
        large_arg = arg > 700
        result = np.empty_like(arg)
        result[large_arg] = np.exp(arg[large_arg]) / (2 * sqrt_ok)
        result[~large_arg] = np.sinh(arg[~large_arg]) / sqrt_ok
        return result
    # Closed universe (Omega_k < 0)
    else:
        sqrt_ok = np.sqrt(-Omega_k)
        return np.sin(sqrt_ok * x) / sqrt_ok

# ----------------------------------------
# 3️⃣ Luminosity distance
# ----------------------------------------
def luminosity_distance_curved(z, Omega_L, Omega_k):
    z_grid = np.linspace(0, np.max(z), 2000)

    Hz = H_of_z_curved(z_grid, Omega_L, Omega_k)
    Hz_SI = Hz.to(1/u.s)
    inv_H = (1 / Hz_SI).value  # seconds
    I_grid = cumtrapz(inv_H, z_grid, initial=0.0)
    I_interp = np.interp(z, z_grid, I_grid)

    # Comoving distance in meters
    D_C = I_interp * u.s * c
    # Luminosity distance with curvature
    D_L = (1 + z) * S_k(D_C.value, Omega_k) * u.m
    return D_L

# ----------------------------------------
# 4️⃣ Apparent magnitude model
# ----------------------------------------
def m_model_curved(z, Omega_L, Omega_k, L):
    """
    Apparent magnitude from luminosity and distance.
    """
    D_L = luminosity_distance_curved(z, Omega_L, Omega_k)
    f = L / (4 * np.pi * D_L**2)  # flux in W/m²/Å
    f_val = np.clip(f.value, 1e-50, None)
    m = -2.5 * np.log10(f_val / f0.value)
    return m










# ----------------------------------------
# 5️⃣ Distance modulus model
# ----------------------------------------
def mu_model_curved(z, Omega_L, Omega_k):
    D_L = luminosity_distance_curved(z, Omega_L, Omega_k)
    D_L_pc = D_L.to(u.pc).value
    mu = 5 * np.log10(D_L_pc / 10.0)
    return mu
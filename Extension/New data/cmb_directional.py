import numpy as np
from scipy.integrate import quad
from scipy.linalg import inv

# ---------------------------------------------------------
# Physical constants
# ---------------------------------------------------------
c = 299792.458  # speed of light [km/s]

# ---------------------------------------------------------
# Wang & Wang (2013) distance prior numbers
# Table II (Planck-based example)
# ---------------------------------------------------------
cmb_mean = np.array([
    1.7499,   # R
    301.46,   # l_A
    0.0224    # Omega_b h^2
])

cmb_cov = np.array([
    [ 1.52e-04,  4.50e-04, -1.20e-05],
    [ 4.50e-04,  2.55e-02, -1.10e-04],
    [-1.20e-05, -1.10e-04,  1.60e-07]
])

cmb_icov = inv(cmb_cov)

# ---------------------------------------------------------
# Eq. (4) Wang & Wang (2013): redshift of last scattering
# ---------------------------------------------------------
def z_star(Omega_m, h, Omega_b_h2=0.0224):
    g1 = (0.0783 * Omega_b_h2**(-0.238)) / \
         (1.0 + 39.5 * Omega_b_h2**0.763)
    g2 = 0.560 / (1.0 + 21.1 * Omega_b_h2**1.81)

    return 1048.0 * (1.0 + 0.00124 * Omega_b_h2**(-0.738)) * \
           (1.0 + g1 * (Omega_m * h**2)**g2)

# ---------------------------------------------------------
# Hubble parameter E(z)
# ---------------------------------------------------------
def E_z(z, Omega_m, Omega_k, Omega_L):
    return np.sqrt(
        Omega_m * (1 + z)**3 +
        Omega_k * (1 + z)**2 +
        Omega_L
    )

# ---------------------------------------------------------
# Transverse comoving distance D_M(z)
# ---------------------------------------------------------
def comoving_distance(z, Omega_m, Omega_k, Omega_L, H0):
    integrand = lambda zp: 1.0 / E_z(zp, Omega_m, Omega_k, Omega_L)
    chi, _ = quad(integrand, 0.0, z)
    Dc = (c / H0) * chi

    if Omega_k > 0:
        return (c / H0) / np.sqrt(Omega_k) * np.sinh(np.sqrt(Omega_k) * H0 * Dc / c)
    elif Omega_k < 0:
        return (c / H0) / np.sqrt(-Omega_k) * np.sin(np.sqrt(-Omega_k) * H0 * Dc / c)
    else:
        return Dc

# ---------------------------------------------------------
# Sound horizon at decoupling (approximate but sufficient)
# ---------------------------------------------------------
def sound_horizon(zs, Omega_m, h, Omega_b_h2=0.0224):
    Omega_r = 4.15e-5 / h**2

    def integrand(z):
        R = 3.0 * Omega_b_h2 / (4.0 * Omega_r * (1 + z))
        return 1.0 / np.sqrt(3.0 * (1.0 + R) *
                              (Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4))

    integral, _ = quad(integrand, zs, np.inf)
    return (c / (100.0 * h)) * integral

# ---------------------------------------------------------
# Wang & Wang (2013) directional CMB prior
# Eq. (9)
# ---------------------------------------------------------
def cmb_directional_logprior(theta):
    Omega_L, Omega_k, H0 = theta
    h = H0 / 100.0
    Omega_m = 1.0 - Omega_L - Omega_k

    # Physical bounds
    if Omega_m <= 0.0 or h <= 0.0:
        return -np.inf

    # z_* from Eq. (4)
    zs = z_star(Omega_m, h)

    # Distances
    DM = comoving_distance(zs, Omega_m, Omega_k, Omega_L, H0)
    rs = sound_horizon(zs, Omega_m, h)

    # Eq. (2): Shift parameter R
    R = np.sqrt(Omega_m * H0**2) * DM / c

    # Eq. (3): Acoustic scale l_A
    lA = np.pi * DM / rs

    # Parameter vector
    p = np.array([R, lA, 0.0224])

    diff = p - cmb_mean
    weaken_factor = 10
    cmb_icov_weak = inv(cmb_cov * weaken_factor**2)

    return -0.5 * diff @ cmb_icov_weak @ diff

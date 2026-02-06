import numpy as np
import matplotlib.pyplot as plt

def unpack(arr):
    return arr[0], -arr[1], arr[2]

dims = np.array([1, 2, 3])
offset = 0.04

# Omega_L data
OL_MCMC = [
    [0.7272, -0.01397, 0.01387],
    [0.7069, -0.01933, 0.01976],
    [0.6865, -0.1116, 0.1107]
]

OL_chi2 = [
    [0.7278, -0.013, 0.013],
    [0.7081, -0.02, 0.019],
    [0.7081, -0.119, 0.105]
]

OL_mcmc_vals = [unpack(x)[0] for x in OL_MCMC]
OL_mcmc_errs = np.array([[unpack(x)[1] for x in OL_MCMC],
                         [unpack(x)[2] for x in OL_MCMC]])

OL_chi2_vals = [unpack(x)[0] for x in OL_chi2]
OL_chi2_errs = np.array([[unpack(x)[1] for x in OL_chi2],
                         [unpack(x)[2] for x in OL_chi2]])

plt.figure(figsize=(7, 6))
plt.errorbar(dims - offset, OL_mcmc_vals, yerr=OL_mcmc_errs,
             fmt='o', label='MCMC', color = 'blue')
plt.errorbar(dims + offset, OL_chi2_vals, yerr=OL_chi2_errs,
             fmt='s', label=r'$\chi^2$', color = 'red')

plt.xticks(dims, dims)
plt.xlabel("Dimensionality", fontsize = 14)
plt.ylabel(r"$\Omega_\Lambda$", fontsize = 14)
plt.title(r"Evolution of $\mathbf{\Omega_\Lambda}$", fontsize = 18, fontweight = 'bold')
plt.legend(loc = 'upper left')
plt.show()

# H0 data
H0_MCMC_2D = [69.54, -0.324, 0.304]
H0_MCMC_3D = [69.44, -0.4115, 0.3836]

H0_chi2_2D = [69.54, -0.306, 0.306]
H0_chi2_3D = [69.54, -0.377, 0.359]

H0_mcmc_vals = [70.0, H0_MCMC_2D[0], H0_MCMC_3D[0]]
H0_mcmc_errs = np.array([
    [0.0, -H0_MCMC_2D[1], -H0_MCMC_3D[1]],
    [0.0,  H0_MCMC_2D[2],  H0_MCMC_3D[2]]
])

H0_chi2_vals = [70.0, H0_chi2_2D[0], H0_chi2_3D[0]]
H0_chi2_errs = np.array([
    [0.0, -H0_chi2_2D[1], -H0_chi2_3D[1]],
    [0.0,  H0_chi2_2D[2],  H0_chi2_3D[2]]
])

plt.figure(figsize=(7, 6))
plt.errorbar(dims - offset, H0_mcmc_vals, yerr=H0_mcmc_errs,
             fmt='o', label='MCMC', color = 'blue')
plt.errorbar(dims + offset, H0_chi2_vals, yerr=H0_chi2_errs,
             fmt='s', label=r'$\chi^2$', color = 'red')

plt.xticks(dims, dims)
plt.xlabel("Dimensionality", fontsize = 14)
plt.ylabel(r"$H_0\ \mathrm{[km\,s^{-1}\,Mpc^{-1}]}$", fontsize = 14)
plt.title(r"Evolution of $\mathbf{H_0}$", fontsize = 18, fontweight = 'bold')
plt.legend(loc = 'lower left')
plt.show()

# Omega_k data (only constrained in 3D)
Ok_MCMC_3D = [0.03512, -0.1808, 0.1734]
Ok_chi2_3D = [0.0001193, -0.166, 0.189]

Ok_mcmc_vals = [0.0, 0.0, Ok_MCMC_3D[0]]
Ok_mcmc_errs = np.array([
    [0.0, 0.0, -Ok_MCMC_3D[1]],
    [0.0, 0.0,  Ok_MCMC_3D[2]]
])

Ok_chi2_vals = [0.0, 0.0, Ok_chi2_3D[0]]
Ok_chi2_errs = np.array([
    [0.0, 0.0, -Ok_chi2_3D[1]],
    [0.0, 0.0,  Ok_chi2_3D[2]]
])

plt.figure(figsize=(7, 6))
plt.errorbar(dims - offset, Ok_mcmc_vals, yerr=Ok_mcmc_errs,
             fmt='o', label='MCMC', color = 'blue')
plt.errorbar(dims + offset, Ok_chi2_vals, yerr=Ok_chi2_errs,
             fmt='s', label=r'$\chi^2$', color = 'red')

plt.xticks(dims, dims)
plt.xlabel("Dimensionality", fontsize = 14)
plt.ylabel(r"$\Omega_k$", fontsize = 14)
plt.title(r"Evolution of $\mathbf{\Omega_k}$", fontsize = 18, fontweight = 'bold')
plt.legend(loc = 'upper left')
plt.show()

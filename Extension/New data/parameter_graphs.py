import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of your script
file_path = os.path.join(script_dir, "CP_results.xlsx")

#order is flat wide/narrow, cmb gaussian/directional
df1D = pd.read_excel(file_path, sheet_name='1D', skiprows=1)
omega_L_1D = df1D["Omega Lambda"].values
omega_L_1D_minus_errors = df1D["OL minus error"].values
omega_L_1D_plus_errors = df1D["OL plus error"].values
df2D = pd.read_excel(file_path, sheet_name='2D', skiprows=1)
omega_L_2D = df2D["Omega Lambda"].values
omega_L_2D_minus_errors = df2D["OL minus error"].values
omega_L_2D_plus_errors = df2D["OL plus error"].values
H0_2D = df2D["H0"].values
H0_2D_minus_errors = df2D["H0 minus error"].values
H0_2D_plus_errors = df2D["H0 plus error"].values
df3D = pd.read_excel(file_path, sheet_name='3D', skiprows=1)
omega_L_3D = df3D["Omega Lambda"].values
omega_L_3D_minus_errors = df3D["OL minus error"].values
omega_L_3D_plus_errors = df3D["OL plus error"].values
H0_3D = df3D["H0"].values
H0_3D_minus_errors = df3D["H0 minus error"].values
H0_3D_plus_errors = df3D["H0 plus error"].values
omega_k_3D = df3D["Omega k"].values
omega_k_3D_minus_errors = df3D["Ok minus error"].values
omega_k_3D_plus_errors = df3D["Ok plus error"].values

dfz = pd.read_excel(file_path, sheet_name='Various zs', skiprows=1)
omega_L_zs = dfz["Omega Lambda"].values
omega_L_zs_minus_errors = dfz["OL minus error"].values
omega_L_zs_plus_errors = dfz["OL plus error"].values
H0_zs = dfz["H0"].values
H0_zs_minus_errors = dfz["H0 minus error"].values
H0_zs_plus_errors = dfz["H0 plus error"].values
omega_k_zs = dfz["Omega k"].values
omega_k_zs_minus_errors = dfz["Ok minus error"].values
omega_k_zs_plus_errors = dfz["Ok plus error"].values

dfchi2 = pd.read_excel(file_path, sheet_name='chi2', skiprows=1)
omega_L_chi2 = dfchi2["Omega Lambda"].values
omega_L_chi2_minus_errors = dfchi2["OL minus error"].values
omega_L_chi2_plus_errors = dfchi2["OL plus error"].values
H0_chi2 = dfchi2["H0"].values
H0_chi2_minus_errors = dfchi2["H0 minus error"].values
H0_chi2_plus_errors = dfchi2["H0 plus error"].values
omega_k_chi2 = dfchi2["Omega k"].values
omega_k_chi2_minus_errors = dfchi2["Ok minus error"].values
omega_k_chi2_plus_errors = dfchi2["Ok plus error"].values
chired_chi2 = [1.7168, 1.716, 1.7184]
MAF = [0.805,0.715,0.646]


#CHI2 VS MCMC
# dims and offset for plotting
dims = np.array([1, 2, 3])
offset = 0.04

def unpack(arr):
    """Return central value and absolute lower/upper errors (all positive)."""
    return arr[0], abs(arr[1]), abs(arr[2])

import numpy as np
import matplotlib.pyplot as plt

def plot_evolution(dims, offset,
                   mcmc_vals, mcmc_errs,
                   chi2_vals, chi2_errs,
                   ylabel, title,
                   literature_val=None,
                   literature_err=None,
                   legend_loc='upper left'):
    plt.figure(figsize=(7, 6))

    # Plot MCMC and chi^2 with asymmetric error bars
    plt.errorbar(dims - offset, mcmc_vals, yerr=mcmc_errs, fmt='o', label='MCMC', color='blue', capsize=4)
    plt.errorbar(dims + offset, chi2_vals, yerr=chi2_errs, fmt='s', label=r'$\chi^2$', color='red', capsize=4)

    # Plot literature central line and dashed error bars
    if literature_val is not None:
        plt.axhline(y=literature_val, color='green', linestyle='-', label='Literature', linewidth=2)
        if literature_err is not None:
            if isinstance(literature_err, (list, tuple, np.ndarray)):
                lower, upper = literature_err
            else:
                lower = upper = literature_err
            plt.axhline(y=literature_val - lower, color='green', linestyle='--')
            plt.axhline(y=literature_val + upper, color='green', linestyle='--')

    plt.xticks(dims, dims)
    plt.xlabel("Dimensionality", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.show()


# --------------------------
# Extract Omega_L data
# --------------------------
OL_MCMC = [
    [omega_L_1D[0], -omega_L_1D_minus_errors[0], omega_L_1D_plus_errors[0]],
    [omega_L_2D[0], -omega_L_2D_minus_errors[0], omega_L_2D_plus_errors[0]],
    [omega_L_3D[0], -omega_L_3D_minus_errors[0], omega_L_3D_plus_errors[0]]
]

OL_chi2 = [
    [omega_L_chi2[0], -omega_L_chi2_minus_errors[0], omega_L_chi2_plus_errors[0]],
    [omega_L_chi2[1], -omega_L_chi2_minus_errors[1], omega_L_chi2_plus_errors[1]],
    [omega_L_chi2[2], -omega_L_chi2_minus_errors[2], omega_L_chi2_plus_errors[2]]
]

OL_mcmc_vals = [unpack(x)[0] for x in OL_MCMC]
OL_mcmc_errs = np.array([[unpack(x)[1] for x in OL_MCMC],
                         [unpack(x)[2] for x in OL_MCMC]])

OL_chi2_vals = [unpack(x)[0] for x in OL_chi2]
OL_chi2_errs = np.array([[unpack(x)[1] for x in OL_chi2],
                         [unpack(x)[2] for x in OL_chi2]])

# --------------------------
# Extract H0 data (2D and 3D only)
# --------------------------
H0_MCMC = [
    [70.0, 0.0, 0.0],  # 1D placeholder
    [H0_2D[0], -H0_2D_minus_errors[0], H0_2D_plus_errors[0]],
    [H0_3D[0], -H0_3D_minus_errors[0], H0_3D_plus_errors[0]]
]

H0_chi2 = [
    [70.0, 0.0, 0.0],
    [H0_chi2[1], -H0_chi2_minus_errors[1], H0_chi2_plus_errors[1]],
    [H0_chi2[2], -H0_chi2_minus_errors[2], H0_chi2_plus_errors[2]]
]

H0_mcmc_vals = [unpack(x)[0] for x in H0_MCMC]
H0_mcmc_errs = np.array([[unpack(x)[1] for x in H0_MCMC],
                         [unpack(x)[2] for x in H0_MCMC]])

H0_chi2_vals = [unpack(x)[0] for x in H0_chi2]
H0_chi2_errs = np.array([[unpack(x)[1] for x in H0_chi2],
                         [unpack(x)[2] for x in H0_chi2]])

# --------------------------
# Extract Omega_k data (only 3D has values)
# --------------------------
Ok_MCMC = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [omega_k_3D[0], -omega_k_3D_minus_errors[0], omega_k_3D_plus_errors[0]]
]

Ok_chi2 = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [omega_k_chi2[2], -omega_k_chi2_minus_errors[2], omega_k_chi2_plus_errors[2]]
]

Ok_mcmc_vals = [unpack(x)[0] for x in Ok_MCMC]
Ok_mcmc_errs = np.array([[unpack(x)[1] for x in Ok_MCMC],
                         [unpack(x)[2] for x in Ok_MCMC]])

Ok_chi2_vals = [unpack(x)[0] for x in Ok_chi2]
Ok_chi2_errs = np.array([[unpack(x)[1] for x in Ok_chi2],
                         [unpack(x)[2] for x in Ok_chi2]])

#plot_evolution(dims, offset, OL_mcmc_vals, OL_mcmc_errs,OL_chi2_vals, OL_chi2_errs,r"$\Omega_\Lambda$",r"Evolution of $\mathbf{\Omega_\Lambda}$",literature_val=0.6847,literature_err=(0.0073, 0.0073),legend_loc='upper left')

#plot_evolution(dims, offset, H0_mcmc_vals, H0_mcmc_errs,H0_chi2_vals, H0_chi2_errs,r"$H_0\ \mathrm{[km\,s^{-1}\,Mpc^{-1}]}$",r"Evolution of $\mathbf{H_0}$",literature_val=67.36,literature_err=(0.54, 0.54),legend_loc='lower left')

#plot_evolution(dims, offset, Ok_mcmc_vals, Ok_mcmc_errs,Ok_chi2_vals, Ok_chi2_errs,r"$\Omega_k$",r"Evolution of $\mathbf{\Omega_k}$",literature_val=0.0007,literature_err=(0.0019, 0.0019),legend_loc='upper left')



def plot_q1_vs_q2_with_errors(
    q1, q1_minus, q1_plus,
    q2, q2_minus, q2_plus,
    xlabel, ylabel
):
    """
    Plot quantity 1 vs quantity 2 with asymmetric error bars.
    Ordering assumed:
    [flat wide, flat narrow, cmb gaussian, cmb directional]
    """

    labels = [
        "Flat wide",
        "Flat narrow",
        "CMB Gaussian",
        "CMB directional"
    ]

    colors = [
        "#A6CEE3",  # light blue
        "#B2DF8A",  # light green
        "#FB9A99",  # light red
        "#FDBF6F"   # light orange
    ]

    markers = ["o", "s", "^", "D"]

    plt.figure(figsize=(6, 6))

    for i in range(4):
        plt.errorbar(
            q1[i],
            q2[i],
            xerr=[[-q1_minus[i]], [q1_plus[i]]],
            yerr=[[-q2_minus[i]], [q2_plus[i]]],
            fmt=markers[i],
            color=colors[i],
            capsize=4,
            markersize=7,
            label=labels[i]
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

plot_q1_vs_q2_with_errors(
    omega_L_3D,
    omega_L_3D_minus_errors,
    omega_L_3D_plus_errors,
    H0_3D,
    H0_3D_minus_errors,
    H0_3D_plus_errors,
    xlabel=r"$\Omega_\Lambda$",
    ylabel=r"$H_0\ \mathrm{[km\,s^{-1}\,Mpc^{-1}]}$"
)
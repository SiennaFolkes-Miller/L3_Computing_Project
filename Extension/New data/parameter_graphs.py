import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 14
TITLE_SIZE = 18

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
    # -------------------------
    # Literature reference line and shaded region
    # -------------------------
    x_left = min(dims) - 0.2
    x_right = max(dims) + 0.2
    if literature_val is not None:
        # Shaded uncertainty band
        if literature_err is not None:
            if isinstance(literature_err, (list, tuple, np.ndarray)):
                lower, upper = literature_err
            else:
                lower = upper = literature_err

            plt.fill_between(
                [x_left, x_right],
                literature_val - lower,
                literature_val + upper,
                color='darkgrey', alpha=0.3
            )
        plt.axhline(literature_val, color='darkgrey', lw=2, label='Literature')

    plt.errorbar(
        dims - offset, mcmc_vals, yerr=mcmc_errs,
        fmt='o', capsize=4, color='teal', label='MCMC'
    )
    plt.errorbar(
        dims + offset, chi2_vals, yerr=chi2_errs,
        fmt='s', capsize=4, color='olive', label=r'$\chi^2$'
    )
    # -------------------------
    # Axes labels, ticks, title, legend
    # -------------------------
    plt.xticks(dims, dims, fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.xlim(x_left, x_right)
    plt.xlabel("Dimensionality", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    plt.title(title, fontsize=TITLE_SIZE)

    plt.legend(loc=legend_loc, fontsize=LEGEND_SIZE)

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

#plot_evolution(dims, offset, Ok_mcmc_vals, Ok_mcmc_errs,Ok_chi2_vals, Ok_chi2_errs,r"$\Omega_k$",r"Evolution of $\mathbf{\Omega_k}$",literature_val=0,literature_err=(0, 0),legend_loc='upper left')





# Plot styling constants
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 14
import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from MCMC import run_supernova_mcmc

# -------------------------
# CONFIGURATION
# -------------------------
priors = ["flat_wide", "flat_narrow", "cmb_gaussian", "cmb_directional"]
colors = ["blue", "green", "red", "orange"]
markers = ["o", "s", "^", "D"]

output_dir = "saved_samples"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
# Replace with your actual z, mu, sigma_mu arrays
from new_data_splitting import z, mu, sigma_mu
import matplotlib.colors as mcolors

posterior_dict = {}
stats_dict = {}

for prior in priors:
    filename = os.path.join(output_dir, f"samples_{prior}.npz")

    if os.path.exists(filename):
        # Load previously saved samples
        data = np.load(filename, allow_pickle=True)
        samples = data["samples"]
        stats = dict(data["stats"].item())
        print(f"Loaded samples for {prior}")
    else:
        # Run MCMC for this prior
        flat_universe = True  # Adjust depending on your prior
        samples, stats = run_supernova_mcmc(
            z, mu, sigma_mu,
            prior_type=prior,
            flat_universe=flat_universe,
            fix_H0=False,
            nwalkers=75,
            nsteps=3000,
            discard=600,
            output=False
        )
        # Save for later
        np.savez(filename, samples=samples, stats=stats)
        print(f"Saved samples for {prior}")

    posterior_dict[prior] = samples
    stats_dict[prior] = stats

# -------------------------
# PLOT 2D CONTOURS
# -------------------------
plt.figure(figsize=(8, 6))

priors = ["flat_narrow", "cmb_gaussian", "cmb_directional"]
labels = ['Flat', 'CMB Gaussian', 'CMB Directional']
colors = ["blue", "orange", "red"]
markers = ["o", "s", "^"]
for i, prior in enumerate(priors):
    samples = posterior_dict[prior]

    # Corner style: get the first two columns (Omega_L, H0)
    x = samples[:, 0]  # Omega_L
    y = samples[:, 1]  # H0

    # Compute 2D histogram for contour
    H, xedges, yedges = np.histogram2d(x, y, bins=100, density=True)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    
    # Levels for 68% credible region
    # Flatten H and compute threshold
    H_flat = H.flatten()
    H_sort = np.sort(H_flat)[::-1]
    cumsum = np.cumsum(H_sort)
    cumsum /= cumsum[-1]
    # Find value corresponding to 68%
    level_68 = H_sort[np.searchsorted(cumsum, 0.68)]
    level_95 = H_sort[np.searchsorted(cumsum, 0.95)]
    base_color = colors[i]
    lighter_color = mcolors.to_rgba(base_color, alpha=0.3)
    
    plt.contour(X, Y, H.T, levels=[level_95], colors=[lighter_color], linewidths=2, linestyles='--')
    plt.contour(X, Y, H.T, levels=[level_68], colors=[base_color], linewidths=2)

    # Plot best-fit point
    best_H0 = stats_dict[prior]["$H_0$"][0]
    best_OL = stats_dict[prior]["$\\Omega_\\Lambda$"][0]
    OL_minus = -omega_L_2D_minus_errors[i]
    OL_plus = omega_L_2D_plus_errors[i]
    H0_minus = -H0_2D_minus_errors[i]
    H0_plus = H0_2D_plus_errors[i]
    plt.errorbar(
        best_OL, best_H0,
        xerr=[[OL_minus], [OL_plus]],
        yerr=[[H0_minus], [H0_plus]],
        fmt=markers[i],
        color=colors[i],
        markersize=8,
        capsize=4,
        label=f"{labels[i]}"
        )

# Optional: literature values
lit_OL = 0.6847
lit_H0 = 67.36
#plt.axvline(lit_OL, color="grey", lw=2, linestyle="--")
#plt.axhline(lit_H0, color="grey", lw=2, linestyle="--")

all_OL = np.concatenate([posterior_dict[p][:,0] for p in priors])
all_H0 = np.concatenate([posterior_dict[p][:,1] for p in priors])
margin_OL = (all_OL.max() - all_OL.min()) * 0.05  # 5% margin
margin_H0 = (all_H0.max() - all_H0.min()) * 0.05
plt.xlim(all_OL.min() - margin_OL, all_OL.max() + margin_OL)
plt.ylim(all_H0.min() - margin_H0, all_H0.max() + margin_H0)

plt.xlabel(r"$\Omega_\Lambda$", fontsize=16)
plt.ylabel(r"$H_0$ [km/s/Mpc]", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
#plt.title("2D posterior contours for different priors", fontsize=16)
plt.tight_layout()
#plt.show()





#literature_val=0.6847,literature_err=(0.0073, 0.0073)
#literature_val=67.36,literature_err=(0.54, 0.54)
#literature_val=0.0007,literature_err=(0.0019, 0.0019)




from Priors_posteriors import log_prior
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# Grid
H0_vals = np.linspace(50, 85, 300)
OmegaL_vals = np.linspace(0.5, 0.9, 300)

H0_grid, OmegaL_grid = np.meshgrid(H0_vals, OmegaL_vals)

prior_types = ["flat_wide", "flat_narrow", "cmb_gaussian", "cmb_directional"]
colors = {
    "flat_wide": "green",
    "flat_narrow": "blue",
    "cmb_gaussian": "orange",
    "cmb_directional": "red"
}

plt.figure(figsize=(8, 6))
ax = plt.gca()

legend_elements = []

# --------------------------------------------------
# 1️⃣ Flat priors (shaded boxes)
# --------------------------------------------------

# flat_wide limits
#rect_fw = Rectangle((0.0, 40.0),  # bottom left (OmegaL, H0)1.5,          # width in OmegaL60.0,         # height in H0 (100 - 40)facecolor=colors["flat_wide"],alpha=0.1)
#ax.add_patch(rect_fw)

#legend_elements.append(Patch(facecolor=colors["flat_wide"], alpha=0.2,label="Flat wide prior"))

# flat_narrow limits
rect_fn = Rectangle(
    (0.6, 60.0),
    0.2,      # 0.8 - 0.6
    15.0,     # 75 - 60
    facecolor=colors["flat_narrow"],
    alpha=0.15
)
ax.add_patch(rect_fn)

legend_elements.append(
    Patch(facecolor=colors["flat_narrow"], alpha=0.3,
          label="Flat")
)

# --------------------------------------------------
# 2️⃣ Gaussian priors (contours)
# --------------------------------------------------
prior_labels = {
    "cmb_gaussian": "CMB Gaussian",
    "cmb_directional": "CMB Directional"
}

for prior in ["cmb_gaussian", "cmb_directional"]:

    logP = np.zeros_like(H0_grid)

    for ix in range(H0_grid.shape[0]):
        for iy in range(H0_grid.shape[1]):
            theta = [OmegaL_grid[ix, iy], 0.0, H0_grid[ix, iy]]
            logP[ix, iy] = log_prior(
                theta,
                prior_type=prior,
                flat_universe=True
            )

    logP -= np.nanmax(logP)
    P = np.exp(logP)
    P /= np.sum(P)

    P_flat = P.flatten()
    P_sort = np.sort(P_flat)[::-1]
    cumsum = np.cumsum(P_sort)
    cumsum /= cumsum[-1]

    level_68 = P_sort[np.searchsorted(cumsum, 0.68)]
    level_95 = P_sort[np.searchsorted(cumsum, 0.95)]

    # 95% dashed
    ax.contour(
        OmegaL_vals,
        H0_vals,
        P.T,
        levels=[level_95],
        colors=[colors[prior]],
        linestyles="--"
    )

    # 68% solid
    ax.contour(
        OmegaL_vals,
        H0_vals,
        P.T,
        levels=[level_68],
        colors=[colors[prior]],
        linewidths=2
    )

    legend_elements.append(
        Line2D([0], [0],
               color=colors[prior],
               lw=2,
               label=f"{prior_labels[prior]} (68%)")
    )

# --------------------------------------------------
# 3️⃣ Final formatting
# --------------------------------------------------

ax.set_xlim(0.55, 0.85)
ax.set_ylim(55, 80)

ax.set_xlabel(r"$\Omega_\Lambda$", fontsize=16)
ax.set_ylabel(r"$H_0$ [km/s/Mpc]", fontsize=16)
ax.tick_params(axis='both', labelsize=14)
#ax.set_title("Visualisation of Priors in $H_0$–$\Omega_\Lambda$ Space")

ax.legend(handles=legend_elements, fontsize=12)

plt.tight_layout()
plt.show()

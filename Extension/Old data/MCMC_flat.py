import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from priors_posterior_flat import log_posterior_m, log_prior_m
from model_flat import m_model

def run_supernova_mcmc_m(
    z,
    m,
    sigma_m,
    Omega_L_init=0.7,
    L_init=3e32,
    nwalkers=32,
    nsteps=100,
    discard=10,
    thin=1,
    plot_chains=True,
    plot_corner=True
):

    ndim = 2

    # Initialize walkers around initial guess
    pos = np.zeros((nwalkers, ndim))
    pos[:, 0] = Omega_L_init + 1e-2 * np.random.randn(nwalkers)
    pos[:, 1] = L_init * (1 + 1e-2 * np.random.randn(nwalkers))

    # Set up sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior_m,
        args=(z, m, sigma_m)
    )

    # Run MCMC
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Flatten chain
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # Compute acceptance fraction
    acceptance_frac = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction: {acceptance_frac:.3f}")
    print(f"Shape of samples: {samples.shape}")

    # Plot walker chains
    if plot_chains:
        fig, axes = plt.subplots(ndim, figsize=(10, 6), sharex=True)
        labels = [r"$\Omega_\Lambda$", r"$L$"]
        for i in range(ndim):
            axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.3)
            axes[i].set_ylabel(labels[i])
        axes[-1].set_xlabel("Step")
        plt.show()

    # Corner plot
    if plot_corner:
        labels = [r"$\Omega_\Lambda$", r"$L$"]
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            #show_titles=True,
            #title_fmt=".2f",
            #title_kwargs={"fontsize":12}
        )
        plt.show()

    param_stats = {}
    for i, name in enumerate(["Omega_Lambda", "L"]):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        err_minus = p50 - p16
        err_plus = p84 - p50
        param_stats[name] = (p50, err_minus, err_plus)
        print(f"{name} = {p50:.4g} -{err_minus:.4g} +{err_plus:.4g}")

    return samples, acceptance_frac, param_stats


from old_data import low_redshift_values, high_redshift_values
data_all = np.vstack([low_redshift_values, high_redshift_values])  
z_m = data_all[:, 0]           
m = data_all[:, 1]           
sigma_m = data_all[:, 2] 

samples, acc, stats = run_supernova_mcmc_m(
    z_m,
    m,
    sigma_m,
    Omega_L_init=0.7,
    L_init=3e32,
    nwalkers=32,
    nsteps=500,
    discard=10,
    thin=1
)


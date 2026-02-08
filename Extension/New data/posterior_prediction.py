# posterior_predictive_run.py
import numpy as np
import matplotlib.pyplot as plt
from models import distance_modulus
from new_data_splitting import z, mu, sigma_mu
from MCMC import run_supernova_mcmc


def posterior_predictive(
    z,
    mu_err,
    samples,
    nsamples=200,
    flat_universe=False,
    fix_H0=False,
    H0_fixed=70.0
):
    """
    Generate posterior predictive realizations.
    """
    idx = np.random.choice(len(samples), nsamples, replace=False)
    mu_pp = []

    for i in idx:
        theta = samples[i]

        j = 0
        Omega_L = theta[j]
        j += 1

        Omega_k = theta[j] if not flat_universe else 0.0
        if not flat_universe:
            j += 1

        H0 = theta[j] if not fix_H0 else H0_fixed

        mu_model = distance_modulus(z, Omega_L, Omega_k, H0)
        mu_real = np.random.normal(mu_model, mu_err)

        mu_pp.append(mu_real)

    return np.array(mu_pp)


def plot_ppc(z, mu_obs, mu_pp):
    mu_med = np.median(mu_pp, axis=0)
    mu_lo = np.percentile(mu_pp, 16, axis=0)
    mu_hi = np.percentile(mu_pp, 84, axis=0)

    plt.figure(figsize=(6, 4))
    plt.fill_between(z, mu_lo, mu_hi, color="lightgray", label="PPC 68%")
    plt.plot(z, mu_med, color="black", label="PPC median")
    plt.scatter(z, mu_obs, s=10, alpha=0.6, label="Observed")

    plt.xlabel("z")
    plt.ylabel(r"$\mu$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    samples, _ = run_supernova_mcmc(
        z, mu, sigma_mu,
        prior_type="flat",
        flat_universe=False,
        fix_H0=False,
        output=False
    )

    mu_pp = posterior_predictive(
        z, sigma_mu, samples,
        flat_universe=False,
        fix_H0=False
    )

    plot_ppc(z, mu, mu_pp)

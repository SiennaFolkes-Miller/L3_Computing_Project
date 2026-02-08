from MCMC import run_supernova_mcmc
from new_data_splitting import z, mu, sigma_mu
import numpy as np
from redshift_variations import run_redshift_scan

#options for prior are flat_wide, flat_narrow, cmb_gaussian, cmb_directional

# 1D: ΩΛ only (flat, H0 fixed)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_wide", flat_universe=True, fix_H0=True, H0_fixed = 70.0, nwalkers=50, nsteps=2000,discard=400,thin=1,output=True)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_narrow", flat_universe=True, fix_H0=True, H0_fixed = 70.0, nwalkers=50, nsteps=2000,discard=400,thin=1,output=True)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="cmb_gaussian", flat_universe=True, fix_H0=True, H0_fixed = 70.0, nwalkers=50, nsteps=2000,discard=400,thin=1,output=True)

# 2D: ΩΛ + H0 (flat)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_wide", flat_universe=True, fix_H0=False, H0_fixed = 70.0, nwalkers=75, nsteps=3000,discard=600,thin=1,output=True)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_narrow", flat_universe=True, fix_H0=False, H0_fixed = 70.0, nwalkers=75, nsteps=3000,discard=600,thin=1,output=True)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="cmb_gaussian", flat_universe=True, fix_H0=False, H0_fixed = 70.0, nwalkers=75, nsteps=3000,discard=600,thin=1,output=True)
# 3D: ΩΛ + Ωk + H0
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_wide", flat_universe=False, fix_H0=False, H0_fixed = 70.0, nwalkers=100, nsteps=5000,discard=1000,thin=1,output=True)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="flat_narrow", flat_universe=False, fix_H0=False, H0_fixed = 70.0, nwalkers=100, nsteps=5000,discard=1000,thin=1,output=True)
run_supernova_mcmc(z, mu, sigma_mu, Omega_L_init = 0.7, Omega_k_init = 0.0, H0_init = 70.0, prior_type="cmb_gaussian", flat_universe=False, fix_H0=False, H0_fixed = 70.0, nwalkers=100, nsteps=5000,discard=1000,thin=1,output=True)


#redshift cuts
zmax_values = np.linspace(0.1, np.max(z), 6)
#run_redshift_scan(zmax_values,prior_type="flat_wide",flat_universe=False,fix_H0=False)
import numpy as np
import astropy.units as u
from astropy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from data_splitting import low_redshift_values
from data_splitting import high_redshift_values
from part2_defs import m_model
from part2_defs import chi_squared
from part2_results import best_Omega_L
from part2_results import average_error
from part2_results import chi2_min
from part2_results import chi2_red
from part2_results import Omega_L_values
from part2_results import chi2_values
from part2_results import sigma_OL_minus
from part2_results import sigma_OL_plus


z_obs = high_redshift_values[:,0]
m_obs = high_redshift_values[:,1]
err = high_redshift_values[:,2]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2,width_ratios=[3, 1],height_ratios=[3, 1],hspace=0,wspace=0)

#data and model fit
ax0 = fig.add_subplot(gs[0, 0]) 
z_model = np.linspace(min(z_obs)*0.9, max(z_obs)*1.1, 200)
m_model_vals = m_model(z_model, best_Omega_L)
ax0.plot(z_model, m_model_vals, color='red', label='Best-fit model')
ax0.errorbar(z_obs, m_obs, yerr=err, color = 'blue', fmt='o', label='Observed', capsize=3)
ax0.set_ylabel('Effective Magnitude (m)',fontsize=14, fontweight='bold', fontname='Arial')
ax0.tick_params(labelbottom=False) 
textstr = (r"$\mathbf{\chi^2_{min}} = \mathbf{" + f"{chi2_min:.2f}" + r"}$" + "\n" r"$\mathbf{\chi^2_{red}} = \mathbf{" + f"{chi2_red:.2f}" + r"}$")
ax0.text(0.05, 0.95, textstr,transform=ax0.transAxes,fontsize=16,verticalalignment='top',horizontalalignment='left',alpha=0.9)

#residuals
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0) 
residuals = np.array((m_obs - m_model(z_obs, best_Omega_L)) / err)
ax1.axhline(y=0, color='black')
ax1.axhline(y=1, color='grey', linestyle='--')
ax1.axhline(y=-1, color='grey', linestyle='--')
ax1.scatter(z_obs, residuals, color='magenta')
ax1.set_xlabel('Redshift (z)',fontsize=14, fontweight='bold', fontname='Arial')
ax1.set_ylabel('Normalised Residuals',fontsize=14, fontweight='bold', fontname='Arial')

#histogram of residuals
ax2 = fig.add_subplot(gs[1, 1], sharey=ax1) 
ax2.hist(residuals, bins=9, color='magenta', orientation='horizontal', density=True)
ax2.axhline(y=0, color='black')
ax2.axhline(y=1, color='grey', linestyle='--')
ax2.axhline(y=-1, color='grey', linestyle='--')
#model and actual Gaussian
mu_resid, sigma_resid = np.mean(residuals), np.std(residuals)
mu_perf, sigma_perf = 0, 1
y = np.linspace(min(residuals), max(residuals), 100)
pdf_resid = norm.pdf(y, mu_resid, sigma_resid)
pdf_perf = norm.pdf(y, mu_perf, sigma_perf)
ax2.plot(pdf_perf, y, color='grey', lw=2)
ax2.plot(pdf_resid, y, color='purple', lw=3)
ax2.set_xlabel('Probability Density',fontsize=14, fontweight='bold', fontname='Arial')
ax2.tick_params(left=False, labelleft=False)  

plt.show()


#chi-squared for different dark energy parameter values
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(Omega_L_values, chi2_values)
ax.hlines(chi2_min, xmin=min(Omega_L_values), xmax=best_Omega_L,color='red', linestyles='-')
ax.hlines(chi2_min + 1, xmin=min(Omega_L_values), xmax=best_Omega_L+sigma_OL_plus,color='red', linestyles='--')
ax.vlines(best_Omega_L, ymin=0, ymax=chi2_min, color='red', linestyles='-')
ax.vlines(best_Omega_L + sigma_OL_plus, ymin=0, ymax=chi2_min+1,color='red', linestyles='--')
ax.vlines(best_Omega_L - sigma_OL_minus, ymin=0, ymax=chi2_min+1,color='red', linestyles='--')
ax.set_xlabel('Dark Energy Density Parameter', fontsize=22, fontweight='bold', fontname='Arial')
ax.set_ylabel(r'Minimised $\mathbf{\chi^2}$', fontsize=22, fontweight='bold', fontname='Arial')
ax.yaxis.set_label_coords(-0.1, 0.7)
ax.set_xlim(0.35,0.7)
ax.set_xticks([best_Omega_L, 0.35, 0.5, 0.65])
ax.set_xticklabels([r'$\mathbf{Best\ \Omega_{\Lambda,0}}$','0.35','0.5','0.65'])
ax.set_ylim(80,86)
ax.set_yticks([chi2_min, chi2_min+1, 80, 85])
ax.set_yticklabels([r'$\mathbf{\chi^2_{min}}$',r'$\mathbf{\chi^2_{min} + 1}$','80','85'])
ax.tick_params(axis='both', labelsize=16) 
textstr = (r"$\mathbf{\chi^2_{min}} = \mathbf{" + f"{chi2_min:.2f}" + r"}$" + "\n" r"$\mathbf{\chi^2_{red}} = \mathbf{" + f"{chi2_red:.2f}" + r"}$")
ax.text(1.8, 2.6, textstr,transform=ax0.transAxes,fontsize=28,verticalalignment='top',horizontalalignment='left',alpha=0.9)
fig.savefig("Poster_OL_chi2.png", dpi=300, bbox_inches='tight')
plt.show()
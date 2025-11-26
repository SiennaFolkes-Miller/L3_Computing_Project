import numpy as np
import astropy.units as u
from astropy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from data_splitting import low_redshift_values
from data_splitting import high_redshift_values
from part1_defs import m_model
from part1_defs import chi_squared
from part1_results import best_L 
from part1_results import average_error
from part1_results import chi2_min
from part1_results import chi2_red
from part1_results import L_values
from part1_results import chi2_values
from part1_results import sigma_L_minus
from part1_results import sigma_L_plus


z_obs = low_redshift_values[:,0]
m_obs = low_redshift_values[:,1]
err = low_redshift_values[:,2]

z_obs2 = high_redshift_values[:,0]
m_obs2 = high_redshift_values[:,1]
err2 = high_redshift_values[:,2]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2,width_ratios=[3, 1],height_ratios=[3, 1],hspace=0,wspace=0)

#data and model fit
ax0 = fig.add_subplot(gs[0, 0]) 
z_model = np.linspace(min(z_obs)*0.9, max(z_obs)*1.1, 200)
m_model_vals = m_model(z_model, best_L)
ax0.plot(z_model, m_model_vals, color='red', label='Best-fit model')
ax0.errorbar(z_obs, m_obs, yerr=err, color = 'blue', fmt='o', label='Observed', capsize=3)
ax0.set_ylabel('Effective Magnitude (m)',fontsize=16, fontweight='bold', fontname='Arial')
ax0.tick_params(labelbottom=False, labelsize=12) 
textstr = (r"$\mathbf{\chi^2_{min}} = \mathbf{" + f"{chi2_min:.2f}" + r"}$" + "\n" r"$\mathbf{\chi^2_{red}} = \mathbf{" + f"{chi2_red:.2f}" + r"}$")
ax0.text(0.05, 0.95, textstr,transform=ax0.transAxes,fontsize=18,verticalalignment='top',horizontalalignment='left',alpha=0.9)

#full data and model fit inset
inset = inset_axes(ax0, width="40%", height="40%", loc='lower right', borderpad=1)
inset.plot(z_model, m_model_vals, color='red')
inset.errorbar(z_obs, m_obs, yerr=err, color = 'blue', fmt='o', markersize=3)
z_model2 = np.linspace(max(z_obs)*1.1, max(z_obs2)*1.1, 200)
m_model_vals2 = m_model(z_model2, best_L)
inset.plot(z_model2, m_model_vals2, color='red', linestyle = '--')
inset.errorbar(z_obs2, m_obs2, yerr=err2, color = 'darkblue', fmt='o', markersize=3)
x_mark = 0.1
inset.axvline(x=x_mark, color='grey', linestyle='-', linewidth=1.5)
inset.set_xticks([])  
inset.set_yticks([])  
inset.set_xticks([0.1])
inset.set_xticklabels([r'$\mathbf{z=0.1}$'])
inset.tick_params(axis='x', labelsize=14) 
inset.xaxis.set_label_position('top') 
inset.xaxis.tick_top() 

#residuals
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0) 
residuals = np.array((m_obs - m_model(z_obs, best_L)) / err)
ax1.axhline(y=0, color='black')
ax1.axhline(y=1, color='grey', linestyle='--')
ax1.axhline(y=-1, color='grey', linestyle='--')
ax1.scatter(z_obs, residuals, color='magenta')
ax1.set_xlabel('Redshift (z)',fontsize=16, fontweight='bold', fontname='Arial')
ax1.set_ylabel('Normalised Residuals',fontsize=16, fontweight='bold', fontname='Arial')
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)

#histogram of residuals
ax2 = fig.add_subplot(gs[1, 1], sharey=ax1) 
ax2.hist(residuals, bins=5, color='magenta', orientation='horizontal', density=True)
ax2.axhline(y=0, color='black')
ax2.axhline(y=1, color='grey', linestyle='--')
ax2.axhline(y=-1, color='grey', linestyle='--')
y = np.linspace(min(residuals), max(residuals), 300)
#model and actual Gaussian
mu_resid, sigma_resid = np.mean(residuals), np.std(residuals)
mu_perf, sigma_perf = 0, 1
y = np.linspace(min(residuals), max(residuals), 100)
pdf_resid = norm.pdf(y, mu_resid, sigma_resid)
pdf_perf = norm.pdf(y, mu_perf, sigma_perf)
ax2.plot(pdf_perf, y, color='grey', lw=2)
ax2.plot(pdf_resid, y, color='purple', lw=3)
ax2.set_xlabel('Probability Density',fontsize=16, fontweight='bold', fontname='Arial')
ax2.tick_params(left=False, labelleft=False, labelsize=12)  
fig.savefig("Poster_L_plot.png", dpi=300, bbox_inches='tight')
plt.show()


#chi-squared for different peak luminosity values
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(np.log(L_values.value), chi2_values)
ax.hlines(chi2_min, xmin=min(np.log(L_values.value)), xmax=np.log(best_L.value),color='red', linestyles='-')
ax.hlines(chi2_min + 1, xmin=min(np.log(L_values.value)), xmax=np.log((best_L+sigma_L_plus).value),color='red', linestyles='--')
ax.vlines(np.log(best_L.value), ymin=0, ymax=chi2_min, color='red', linestyles='-')
ax.vlines(np.log((best_L + sigma_L_plus).value), ymin=0, ymax=chi2_min+1,color='red', linestyles='--')
ax.vlines(np.log((best_L - sigma_L_minus).value), ymin=0, ymax=chi2_min+1,color='red', linestyles='--')
ax.set_xlabel(r'$\mathbf{log_{10}}$ Peak Luminosity', fontsize=14, fontweight='bold', fontname='Arial')
ax.set_ylabel(r'Minimised $\mathbf{\chi^2}$', fontsize=14, fontweight='bold', fontname='Arial')
ax.set_xlim(74.7,75)
#ax.set_xticks([np.log(best_L.value), np.log((best_L + sigma_L_plus).value),np.log((best_L - sigma_L_minus).value),74.5,74.7,74.9])
#ax.set_xticklabels(['$Best L$','$L+err$','$L-err$','74.5','74.7','74.9'])
ax.set_xticks([np.log(best_L.value),74.7,74.8,74.9,75.0])
ax.set_xticklabels([r'$\mathbf{Best\ L_{peak}}$','74.7','74.8','74.9','75.0'])
ax.set_ylim(20,30)
ax.set_yticks([chi2_min, chi2_min+1,20,23,26,29])
ax.set_yticklabels([r'$\mathbf{\chi^2_{min}}$',r'$\mathbf{\chi^2_{min}+1}$','20','23','26','29'])
textstr = (r"$\mathbf{\chi^2_{min}} = \mathbf{" + f"{chi2_min:.2f}" + r"}$" + "\n" r"$\mathbf{\chi^2_{red}} = \mathbf{" + f"{chi2_red:.2f}" + r"}$")
ax.text(0.6, 0.8, textstr,transform=ax0.transAxes,fontsize=16,verticalalignment='top',horizontalalignment='left',alpha=0.9)
plt.show()
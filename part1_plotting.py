import numpy as np
import astropy.units as u
from astropy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from data_splitting import low_redshift_values
from data_splitting import high_redshift_values
from part1_defs import m_model
from part1_defs import chi_squared
from part1_results import best_L 
from part1_results import average_error
from part1_results import chi2_min
from part1_results import chi2_red

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
ax0.set_ylabel('Effective magnitude (m)')
ax0.tick_params(labelbottom=False) 
textstr = f"$\\chi^2_{{min}} = {chi2_min:.2f}$\n$\\chi^2_{{red}} = {chi2_red:.2f}$"
ax0.text(0.05, 0.95, textstr, transform=ax0.transAxes,fontsize=10, verticalalignment='top', horizontalalignment='left', alpha=0.8)

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
inset.set_xticklabels(['z=0.1'])
inset.xaxis.set_label_position('top') 
inset.xaxis.tick_top() 

#residuals
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0) 
residuals = np.array((m_obs - m_model(z_obs, best_L)) / err)
ax1.axhline(y=0, color='black')
ax1.axhline(y=1, color='grey', linestyle='--')
ax1.axhline(y=-1, color='grey', linestyle='--')
ax1.scatter(z_obs, residuals, color='magenta')
ax1.set_xlabel('Redshift (z)')
ax1.set_ylabel('Normalised residuals')

#histogram of residuals
ax2 = fig.add_subplot(gs[1, 1], sharey=ax1) 
ax2.hist(residuals, bins=7, color='magenta', orientation='horizontal')
ax2.axhline(y=0, color='black')
ax2.axhline(y=1, color='grey', linestyle='--')
ax2.axhline(y=-1, color='grey', linestyle='--')
ax2.set_xlabel('Count')
ax2.tick_params(left=False, labelleft=False)  

plt.show()

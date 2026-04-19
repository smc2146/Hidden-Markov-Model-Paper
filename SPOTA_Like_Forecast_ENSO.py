#==============================================================================
#
# SPOTA-like forcast using the UBMN model - ENSO
#
#==============================================================================


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


# Local imports
import Forecast_Models as smodel
import Forecast_Plots as splot


# Local variables
DEBUG = 1  # Debug level (0=None, 1=Some, 2=More, ...)


# --- Get Datasets ------------------------------------------------------------


# Print header
print('\nRead data')        


# Read data into dataframe
data_df = pd.read_csv(r'data\QLD_Apr_Mar_SOI_Jun_Mar_TPI_Jul_Jun.csv', comment='#')


# Get time data
t = data_df['Year'].to_numpy()
t_num = t.shape[0]


# Get rain data
x = data_df['QLD_Apr_Mar'].to_numpy()
splot.make_time_plot(t, x)
splot.make_dist_plot(x)


# Get state data (2-state based on P50)
s = np.zeros((t_num), dtype=int)
if False:
    ns = 2
    x_q1 = np.median(x)
    print('x_q1 =', x_q1)
    for i in range(t_num):
        if x[i] < x_q1:
            s[i] = 0
        else:
            s[i] = 1
        # end if
    # end for


# Get state data (3-state based on P33, P66)
elif True:
    ns = 3
    x_q1 = np.quantile(x, 0.33)
    x_q2 = np.quantile(x, 0.67)
    print('x_q1 =', x_q1)
    print('x_q2 =', x_q2)
    for i in range(t_num):
        if x[i] < x_q1:
            s[i] = 0
        elif x[i] < x_q2:
            s[i] = 1
        else:
            s[i] = 2
        # end if
    # end for


# Get state data (2-state based on ENSO)
elif False:
    # 0 = El Nino, 1 = La Nina
    ns = 2
    for i in range(t_num):
        if data_df['SOI_Jun_Mar'].iloc[i] < 0.0:
            s[i] = 0
        else:
            s[i] = 1
        # end if
    # end for


# Get state data (3-state based on ENSO)
elif True:
    # 0 = El Nino, 1 = Neutral, 2 = La Nina
    ns = 3
    for i in range(t_num):
        if data_df['SOI_Jun_Mar'].iloc[i] < -3.5:
            s[i] = 0
        elif data_df['SOI_Jun_Mar'].iloc[i] < 4.5:
            s[i] = 1
        else:
            s[i] = 2
        # end if
    # end for
   
    
# end if
print('\nStates:')
print('  ns =', ns)
print('  s:\n', s)


# --- Run Forecast Model ------------------------------------------------------


# Print header
print('\nTesting UBNM model')        


# Fit model
m1 = smodel.Model_UBMN()
m1.fit(x, ns, s)
m1.print()


# Set forecast variables
r1_num = 10000
t1_num = 10
t1 = np.arange(1.0, 1.0 + float(t1_num), 1.0)
t0 = 0.0


# Sample from model (previous state specified)
# s0 = 0
# s0 = 1
s0 = s[-1]  # Last observed state 
x1, s1 = m1.sample(r1_num, t1_num, s0=s0, seed=1)


# Make outputs
splot.make_dist_plot(x, x1)
splot.make_time_plot(t1, x[0:10], xstoc=x1)


# --- SPOTA-like Analysis -----------------------------------------------------


print('\nCheck out distributions')

# Get sample of first time step
y1 = np.zeros((r1_num), dtype=float)
y1[:] = x1[0,:]


# --- KDE ---


# Fit obs KDE
obs_kde_pdf = gaussian_kde(x)
obs_kde_cdf = np.vectorize(obs_kde_pdf.integrate_box_1d)


# Set up obs inv cdf
obs_x = np.linspace(0.0, 1.5*np.max(x), num=1000)
obs_pdf = obs_kde_pdf(obs_x)
obs_cdf = obs_kde_cdf(0.0, obs_x)
obs_kde_inv = interp1d(obs_cdf, obs_x)


# Fit fore KDE
fore_kde_pdf = gaussian_kde(y1)
fore_kde_cdf = np.vectorize(fore_kde_pdf.integrate_box_1d)


# Set up obs inv cdf
fore_x = np.linspace(0.0, 1.5*np.max(y1), num=1000)
fore_pdf = fore_kde_pdf(fore_x)
fore_cdf = fore_kde_cdf(0.0, fore_x)
fore_kde_inv = interp1d(fore_cdf, fore_x)


# Make pdf chart
fig, ax = plt.subplots()
ax.plot(obs_x, obs_pdf, color='tab:blue', label='Obs.')
ax.plot(fore_x, fore_pdf, color='tab:orange', label='Fore.')
plt.legend()
plt.tight_layout()
plt.show()


# Make cdf chart
fig, ax = plt.subplots()
ax.plot(obs_x, obs_cdf, color='tab:blue', label='Obs.')
ax.plot(fore_x, fore_cdf, color='tab:orange', label='Fore.')
plt.legend()
plt.tight_layout()
plt.show()


# --- SPOTA Catagories


# Calculate obs categories
obs_cats = [0.3, 0.4, 0.3]
obs_t1 = obs_kde_inv(0.3)
obs_t2 = obs_kde_inv(0.7)
print(obs_t1, obs_t2)


# Calculate fore categories
fore_cats = []
fore_cats.append(fore_kde_cdf(0.0, obs_t1))
fore_cats.append(fore_kde_cdf(0.0, obs_t2)-fore_kde_cdf(0.0, obs_t1))
fore_cats.append(1.0 - fore_kde_cdf(0.0, obs_t2))
print(fore_cats)


# Make pie chart
fig, axs = plt.subplots(1,2)
axs[0].set_title('Reference')
axs[0].pie(obs_cats, labels=['Dry', 'Average', 'Wet'],
         colors=['tab:orange', 'tab:green','tab:blue'],
         startangle=90.0, autopct = '%1.1f%%',
         wedgeprops = {'edgecolor' : 'black', 'linewidth': 1, 'antialiased': True})
axs[1].set_title('Forecast 2025/26')
axs[1].pie(fore_cats, labels=['Dry', 'Average', 'Wet'],
         colors=['tab:orange', 'tab:green','tab:blue'],
         startangle=90.0, autopct = '%1.1f%%',
         wedgeprops = {'edgecolor' : 'black', 'linewidth': 1, 'antialiased': True})
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------


# Write footer
print('\nCompleted')


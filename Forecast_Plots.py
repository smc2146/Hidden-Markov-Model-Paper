#==============================================================================
#
# Forecast - Plotting functions
#
#------------------------------------------------------------------------------
#
# Contains:
#
#   make_time_plot - Make time plot
#   make_dist_plot - Make distribution plot
#   make_acf_plot  - Make autocorrelation plot
#   make_par_plot  - make parameter pair-wise plot
#   fade           - Fade colour (to white) by a percentage
#
#==============================================================================


# Imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf


# Local imports
from Forecast_Common import report_error, report_warning


# Local variables
DEBUG = 0  # Debug level: 0=None, 1=Some, 2=More, ...




#------------------------------------------------------------------------------
#
# Make time plot
#
#------------------------------------------------------------------------------
def make_time_plot(t, x, xstoc=None, t0=None, x0=None,
                   out_yscale='linear', out_title='Time plot', out_file=None):

    """
    Create time-plot for stochastic data
    
    Parameters
    ----------
    t : ndarray
        1D float array containing time data.
    x : ndarray
        1D float array containing historical value data.
    xstoc : ndarray, optional
        2D float array containing stochastic value data. The default is None.
    t0 : float, optional
        Initial time point. The default is None.
    x0 : float, optional
        Initial value point. The default is None.
    out_yscale : str, optional
        Type of y-axis (linear or log). The default is 'linear'.
    out_title : str, optional
        Title of plot. The default is 'Time plot'.
    out_file : str, optional
        Save plot to file. The default is None.
    
    Returns
    -------
    none
    
    """

    # If required prepend with t0 and x0
    if (t0 is None and not x0 is None) or (not t0 is None and x0 is None):
        report_error('t0 and x0 must both be either none or a value')
    # end if
    if t0 is None:
        t_ = t
        x_ = x
        if xstoc is None:
            xstoc_ = None
        else:
            xstoc_ = xstoc
        # end if
    else:
        t_ = np.insert(t, 0, t0, axis=0)
        x_ = np.insert(x, 0, x0, axis=0)
        if xstoc is None:
            xstoc_ = None
        else:
            xstoc_ = np.insert(xstoc, 0, x0, axis=0)
        # end if
    # end if
        
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle(out_title, fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid()
    # ----------
    ax.plot(t_, x_, color='tab:blue', lw=1.0,
            marker='o', ms=3.0, mec='tab:blue', mfc=fade('tab:blue', 75.0),
            label='x')
    # ----------
    if not xstoc_ is None:
        xstoc_q1 = np.quantile(xstoc_, norm.cdf(-2.0), axis=1)
        xstoc_q2 = np.quantile(xstoc_, norm.cdf(-1.0), axis=1)
        xstoc_q3 = np.quantile(xstoc_, norm.cdf( 0.0), axis=1)
        xstoc_q4 = np.quantile(xstoc_, norm.cdf( 1.0), axis=1)
        xstoc_q5 = np.quantile(xstoc_, norm.cdf( 2.0), axis=1)
        ax.fill_between(t_, xstoc_q1, xstoc_q5,
                        fc=fade('tab:orange', 90.0), ec=None, label='x stoc.($\pm2\sigma$)')
        ax.fill_between(t_, xstoc_q2, xstoc_q4,
                        fc=fade('tab:orange', 70.0), ec=None, label='x stoc.($\pm1\sigma$)')
        ax.plot(t_, xstoc_q3, color='tab:orange', lw=1.0, label='x stoc.(median)')
    # end if
    # ----------
    ax.set_yscale(out_yscale)
    fig.legend(loc='upper center', frameon=False,  ncol=10, 
                bbox_to_anchor=(0.5, 0.95))
  
    # Display or save plot
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=600, format='png', bbox_inches='tight')
    # end if
    plt.close() 

# end def




#------------------------------------------------------------------------------
# Make distribution plot
#   Optionally plot stochastic data (xstoc[num,reps])
#   Optionally set plotting position alpha (out_alpha)
#   Optionally set yscale (out_yscale)
#   Optionally set plot title (out_title)
#   Optionally plot 1:1 line (out_11_line)
#   Optionally save plot to file (out_file)
#------------------------------------------------------------------------------
def make_dist_plot(x, xstoc=None, out_alpha=0.4, out_yscale='linear',
                   out_title='Distribution plot', out_11_line=False,
                   out_file=None):

    # Create plotting position data
    x_num = x.shape[0]
    x_p = (np.arange(1.0, x_num + 1.0) - out_alpha) / (x_num + 1.0 - 2.0*out_alpha)
    x_z = norm.ppf(x_p)

    # If required, create stochastic plotting data
    if not xstoc is None:
        xstoc_num = xstoc.shape[0]
        xstoc_p = (np.arange(1.0, xstoc_num + 1.0) - out_alpha) / (xstoc_num + 1.0 - 2.0*out_alpha)
        xstoc_z = norm.ppf(xstoc_p)
        xstoc_s = np.sort(xstoc, axis=0)
        xstoc_q1 = np.quantile(xstoc_s, norm.cdf(-2.0), axis=1)
        xstoc_q2 = np.quantile(xstoc_s, norm.cdf(-1.0), axis=1)
        xstoc_q3 = np.quantile(xstoc_s, norm.cdf( 0.0), axis=1)
        xstoc_q4 = np.quantile(xstoc_s, norm.cdf( 1.0), axis=1)
        xstoc_q5 = np.quantile(xstoc_s, norm.cdf( 2.0), axis=1)
    # end if

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle(out_title, fontweight='bold')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.grid()
    ax.axvline(x=0.0, color='black', ls='-', lw=0.75)
    # ----------
    if out_11_line:
        x_min = min(np.min(x), np.min(x_z))
        x_max = max(np.max(x), np.max(x_z))
        ax.plot([x_min, x_max], [x_min, x_max], color='tab:purple', lw=1.0, label='1:1 Line')
    # end if
    # ----------
    if not xstoc is None:
        ax.fill_between(xstoc_z, xstoc_q1, xstoc_q5,
                        fc=fade('tab:orange', 90.0), ec=None, label='x stoc.($\pm2\sigma$)')
        ax.fill_between(xstoc_z, xstoc_q2, xstoc_q4,
                        fc=fade('tab:orange', 70.0), ec=None, label='x stoc.($\pm1\sigma$)')
        ax.plot(xstoc_z, xstoc_q3, color='tab:orange', lw=1.0, label='x stoc.(median)')
    # end if
    # ----------
    ax.plot(x_z, np.sort(x), color='tab:blue', lw=1.0,
            marker='o', ms=3.0, mec='tab:blue', mfc=fade('tab:blue', 75.0),
            label='x')
    # ----------
    ax.set_yscale(out_yscale)
    fig.legend(loc='upper center', frameon=False,  ncol=10, 
                bbox_to_anchor=(0.5, 0.95))
  
    # Display or save plot
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=600, format='png', bbox_inches='tight')
    # end if
    plt.close() 

# end def




#------------------------------------------------------------------------------
# Make autocorrelation plot
#   Optionally set no. lags (out_lags)
#   Optionally set significance level (out_alpha)
#   Optionally calculate PACF rather than ACF (out_pacf)
#   Optionally set plot title (out_title)
#   Optionally save plot to file (out_file)
#------------------------------------------------------------------------------
def make_acf_plot(x, out_lags=10, out_alpha=0.05, out_pacf=False,
                  out_title='ACF plot', out_file=None):

    # Calculate ACF
    x_num = x.shape[0]
    x_lags = np.arange(out_lags+1)
    if out_pacf:
        x_acf, x_acf_conf = pacf(x, nlags=out_lags, alpha=out_alpha)
    else:
        x_acf, x_acf_conf = acf(x, nlags=out_lags, alpha=out_alpha)
    # end if
        

    # Create significant lags limit
    sig_lags = np.arange(out_lags+0.1, step=0.1)
    sig_z = norm.ppf(1.0 - out_alpha / 2.0)
    sig_neg = (-1.0 - sig_z * (x_num - sig_lags - 1.0) ** 0.5) / (x_num - sig_lags)
    sig_pos = (-1.0 + sig_z * (x_num - sig_lags - 1.0) ** 0.5) / (x_num - sig_lags)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle(out_title, fontweight='bold')
    ax.set_xlabel('lag')
    ax.set_ylabel('acf')
    # ----------
    ax.grid()
    ax.axhline(y=0.0, color='black', ls='-', lw=0.75)
    ax.plot(sig_lags, sig_neg, c='tab:purple', lw=1.0, label='sig. lag (95%)')
    ax.plot(sig_lags, sig_pos, c='tab:purple', lw=1.0)
    ax.plot(x_lags, x_acf, color='tab:blue', lw=1.0,
            marker='o', ms=3.0, mec='tab:blue', mfc=fade('tab:blue', 75.0),
            label='acf')
    ax.fill_between(x_lags, x_acf_conf[:,0], x_acf_conf[:,1],
                      fc='tab:blue', alpha=0.15, ec=None, label='acf (conf)')
    # ----------
    ax.set_xlim((0.0, None))
    ax.set_ylim((-1.0, 1.0))
    fig.legend(loc='upper center', frameon=False,  ncol=10, 
                bbox_to_anchor=(0.5, 0.95))
  
    # Display or save plot
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=600, format='png', bbox_inches='tight')
    # end if
    plt.close()

# end def




#------------------------------------------------------------------------------
# Make parameter pair-wise plot
#   Optionally plot historical parameter values (phist)
#   Optionally set parameter names (names)
#   Optionally set plot title (out_title)
#   Optionally save plot to file (out_file)
#------------------------------------------------------------------------------
def make_par_plot(p, phist=None, names=None, out_title='Pair plot',
                  out_file=None):

    # Create dataframe
    p_df = pd.DataFrame(np.transpose(p))
    if not names is None:
        p_df.columns = names

    # Create plot
    g = sns.PairGrid(p_df, diag_sharey=False)
    g.fig.subplots_adjust(top=0.94)
    g.fig.suptitle(out_title, fontweight='bold')
    g.map_lower(sns.kdeplot, color='tab:orange')
    g.map_diag(sns.histplot, kde=True, color='tab:orange')
    g.map_upper(sns.scatterplot, color='tab:orange')
    # ----------
    if not phist is None:
        for i in range(4):
            for j in range(4):
                if i == j:
                    g.axes[i][j].axvline(phist[i], c='tab:blue')
                else:
                    g.axes[i][j].scatter(phist[j], phist[i], c='tab:blue')
                # end if
            # end for
        # end for
    # end if

    # Display or save plot
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=600, format='png', bbox_inches='tight')
    # end if
    plt.close()

# end def




#------------------------------------------------------------------------------
# Fade colour (to white) by a percentage
#------------------------------------------------------------------------------
def fade(colour, percent):
    r = percent / 100.0
    c = (1.0 - r) * mcolors.to_rgba_array(colour) + r * mcolors.to_rgba_array('white')
    return (c[0][0], c[0][1], c[0][2])
# end def

